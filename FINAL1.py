# Enahcned Diabetc Tongue Dectetion Sytsem
import os
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall
from PIL import Image
import json
import shutil
from pathlib import Path
import matplotlib.pyplot as plt

# Set environemnt varible to handle OpenMP issue
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class DiabetesTongueDetector:
    def __init__(self):
        self.model = None
        self.class_labels = ["Diabetic", "Healthy"]
        self.img_size = (128, 128)
        self.input_shape = (128, 128, 3)
        
    def create_model(self):
        """Cerate and compiel the CNN modle"""
        model = Sequential([
            # Firts convolutinoal block
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            
            # Secnd convoultional bolck
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            
            # Thrid convoluiontal block
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            
            # Fouth convolutiomal block
            Conv2D(256, kernel_size=(3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            
            # Dropuot for regularizaton
            Dropout(0.3),
            
            # Flaten and dnse layres
            Flatten(),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(), Recall()]
        )
        
        return model
    
    def prepare_data_generators(self, train_dir, validation_split=0.2):
        """Prepare data generator with augmnetation"""
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        validation_datagen = ImageDataGenerator(
            rescale=1./255
        )
        
        train_generator = train_datagen.flow_from_directory(
            os.path.join(train_dir, 'train'),
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical',
            shuffle=True
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            os.path.join(train_dir, 'valid'),
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical',
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    def setup_dataset_structure(self):
        """Setu dataset direcotry strucure"""
        dataset_dir = Path('dataset')
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Crete train, validation, and test directries
        for split in ['train', 'valid', 'test']:
            split_dir = dataset_dir / split
            split_dir.mkdir(exist_ok=True)
            
            # Creat class directries
            diabetic_dir = split_dir / 'diabetic'
            healthy_dir = split_dir / 'healthy'
            diabetic_dir.mkdir(exist_ok=True)
            healthy_dir.mkdir(exist_ok=True)
            
            # Cpy diabetic images from preprocessedcropped
            source_diabetic = Path(f'diabetes(toungue)/preprocessedcropped/{split}/diabetes')
            if source_diabetic.exists():
                for img_file in source_diabetic.glob('*.jpg'):
                    dst = diabetic_dir / img_file.name
                    if not dst.exists():
                        shutil.copy2(img_file, dst)
            
            # Cpy non-diabetic images from preprocessedcropped
            source_nondiabetic = Path(f'diabetes(toungue)/preprocessedcropped/{split}/nondiabetes')
            if source_nondiabetic.exists():
                for img_file in source_nondiabetic.glob('*.jpg'):
                    dst = healthy_dir / img_file.name
                    if not dst.exists():
                        shutil.copy2(img_file, dst)
            
            # If no images were copid from preprocessedcropped, try the Helthy direcotry
            if not any(healthy_dir.glob('*.jpg')):
                source_healthy = Path('Healthy')
                if source_healthy.exists():
                    for img_file in source_healthy.glob('*.jpg'):
                        dst = healthy_dir / img_file.name
                        if not dst.exists():
                            shutil.copy2(img_file, dst)
        
        return str(dataset_dir)
    
    def train_model(self, epochs=30, patience=5):
        """Train the model with earaly stoping and learing rate reducion"""
        try:
            # Setup datset
            dataset_path = self.setup_dataset_structure()
            
            # Chekc if dataset exits
            if not os.path.exists(os.path.join(dataset_path, 'train')):
                raise ValueError("Dataset direcotry not foud. Please ensur the data direcotries exsit.")
            
            # Creat modle
            self.model = self.create_model()
            
            # Preapre data genetarors
            train_gen, val_gen = self.prepare_data_generators(dataset_path)
            
            # Calbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Train modle
            history = self.model.fit(
                train_gen,
                steps_per_epoch=max(1, train_gen.samples // train_gen.batch_size),
                epochs=epochs,
                validation_data=val_gen,
                validation_steps=max(1, val_gen.samples // val_gen.batch_size),
                callbacks=callbacks,
                verbose=1
            )
            
            # Clean up
            if os.path.exists('dataset'):
                shutil.rmtree('dataset')
            
            return history
            
        except Exception as e:
            # Clean up on erro
            if os.path.exists('dataset'):
                shutil.rmtree('dataset')
            raise e
    
    def save_model(self, model_path='model.json', weights_path='model.h5'):
        """Save the trainied modle"""
        if self.model is None:
            raise ValueError("No modle to save. Train the modle first.")
        
        # Save modle architecure
        model_json = self.model.to_json()
        with open(model_path, 'w') as json_file:
            json_file.write(model_json)
        
        # Save modle weigts
        self.model.save_weights(weights_path)
    
    def load_model(self, model_path='model.json', weights_path='model.h5'):
        """Load a trainied modle"""
        try:
            with open(model_path, 'r') as json_file:
                loaded_model_json = json_file.read()
            
            self.model = model_from_json(loaded_model_json)
            self.model.load_weights(weights_path)
            
            # Compile the loaded modle
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy', Precision(), Recall()]
            )
            
            return True
        except Exception as e:
            st.error(f"Error loading modle: {str(e)}")
            return False
    
    def predict_image(self, image_path):
        """Predict the clas of an image"""
        if self.model is None:
            raise ValueError("No modle loaded. Train or load a modle first.")
        
        # Load and preporcess image
        test_image = image.load_img(image_path, target_size=self.img_size)
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255.0
        test_image = np.expand_dims(test_image, axis=0)
        
        # Make predictio
        prediction = self.model.predict(test_image, verbose=0)
        predicted_class = self.class_labels[np.argmax(prediction)]
        confidence = prediction[0]
        
        return predicted_class, confidence

# Streamlit App
def main():
    # Page configuration
    st.set_page_config(
        page_title="Enhanced Diabetic Tongue Detection",
        page_icon="👅",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            margin-top: 1rem;
        }
        .prediction-box {
            padding: 1.5rem;
            border-radius: 10px;
            margin-top: 1rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .healthy {
            background: linear-gradient(135deg, #d4f1d4, #a8e6a3);
            border-left: 5px solid #28a745;
        }
        .diabetic {
            background: linear-gradient(135deg, #ffe6e6, #ffb3b3);
            border-left: 5px solid #dc3545;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = DiabetesTongueDetector()
    
    # Title and description
    st.title("👅 Enhanced Diabetic Tongue Detection System")
    st.markdown("""
    ### Advanced AI-Powered Medical Diagnosis Tool
    This application uses a sophisticated Convolutional Neural Network (CNN) to analyze tongue images 
    and detect potential signs of diabetes. The model incorporates state-of-the-art deep learning 
    techniques for accurate medical image classification.
    """)
    
    # Sidebar
    with st.sidebar:
        st.title("🔧 Controls")
        st.markdown("---")
        
        # Model information
        st.subheader("📊 Model Information")
        if os.path.exists('model.json') and os.path.exists('model.h5'):
            st.success("✅ Trained model available")
        else:
            st.warning("⚠️ No trained model found")
        
        st.markdown("---")
        
        # Training parameters
        st.subheader("🎛️ Training Parameters")
        epochs = st.slider("Epochs", min_value=10, max_value=100, value=30)
        patience = st.slider("Early Stopping Patience", min_value=3, max_value=15, value=5)
        
        st.markdown("---")
        
        # About section
        st.subheader("ℹ️ About")
        st.info("""
        **Technology Stack:**
        - TensorFlow/Keras
        - Convolutional Neural Networks
        - Data Augmentation
        - Transfer Learning Principles
        
        **Features:**
        - Early Stopping
        - Learning Rate Reduction
        - Batch Normalization
        - Dropout Regularization
        """)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Image Classification", "🏋️ Model Training", "📈 Model Analytics"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📤 Upload Tongue Image")
            uploaded_file = st.file_uploader(
                "Choose a tongue image...", 
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image of a tongue for analysis"
            )
            
            if uploaded_file is not None:
                image_pil = Image.open(uploaded_file)
                st.image(image_pil, caption="Uploaded Image", use_column_width=True)
                
                # Save temporarily
                temp_path = "temp_image.jpg"
                image_pil.save(temp_path)
                
                # Image properties
                st.markdown("**Image Properties:**")
                st.write(f"- Size: {image_pil.size}")
                st.write(f"- Mode: {image_pil.mode}")
                st.write(f"- Format: {uploaded_file.type}")
        
        with col2:
            st.subheader("🔮 Prediction Results")
            
            if uploaded_file is not None:
                if st.button("🚀 Classify Image", type="primary"):
                    try:
                        # Load model if not already loaded
                        if st.session_state.detector.model is None:
                            with st.spinner("Loading model..."):
                                if not st.session_state.detector.load_model():
                                    st.error("Please train the model first!")
                                    st.stop()
                        
                        # Make prediction
                        with st.spinner("Analyzing image..."):
                            prediction, confidence = st.session_state.detector.predict_image(temp_path)
                        
                        # Display results
                        css_class = "healthy" if prediction == "Healthy" else "diabetic"
                        
                        st.markdown(f"""
                        <div class="prediction-box {css_class}">
                            <h2>{"🟢" if prediction == "Healthy" else "🔴"} {prediction}</h2>
                            <h4>Confidence Scores:</h4>
                            <p><strong>Diabetic:</strong> {confidence[0]:.2%}</p>
                            <p><strong>Healthy:</strong> {confidence[1]:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence visualization
                        fig, ax = plt.subplots(figsize=(8, 4))
                        bars = ax.bar(['Diabetic', 'Healthy'], confidence, 
                                    color=['#ff6b6b', '#51cf66'])
                        ax.set_ylabel('Confidence')
                        ax.set_title('Prediction Confidence')
                        ax.set_ylim(0, 1)
                        
                        # Add value labels on bars
                        for bar, conf in zip(bars, confidence):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{conf:.2%}', ha='center', va='bottom')
                        
                        st.pyplot(fig)
                        
                        # Medical disclaimer
                        st.warning("""
                        ⚠️ **Medical Disclaimer**: This tool is for educational purposes only. 
                        Always consult with healthcare professionals for medical diagnosis and treatment.
                        """)
                        
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
    
    with tab2:
        st.subheader("🏋️ Model Training")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Training Requirements:**
            - Ensure you have 'Diabetes' and 'Healthy' folders with tongue images
            - Images should be in JPG format
            - Minimum 50 images per class recommended for good performance
            """)
            
            # Dataset info
            diabetes_count = len([f for f in os.listdir('Diabetes')] if os.path.exists('Diabetes') else [])
            healthy_count = len([f for f in os.listdir('Healthy')] if os.path.exists('Healthy') else [])
            
            st.markdown(f"""
            **Current Dataset:**
            - Diabetic images: {diabetes_count}
            - Healthy images: {healthy_count}
            - Total: {diabetes_count + healthy_count}
            """)
        
        with col2:
            if st.button("🚀 Start Training", type="primary"):
                if diabetes_count == 0 or healthy_count == 0:
                    st.error("Please ensure both 'Diabetes' and 'Healthy' folders exist with images!")
                else:
                    try:
                        with st.spinner("Training model... This may take several minutes."):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # Train model
                            history = st.session_state.detector.train_model(epochs=epochs, patience=patience)
                            
                            # Save model
                            st.session_state.detector.save_model()
                            
                            progress_bar.progress(100)
                            status_text.text("Training completed!")
                            
                            st.success("✅ Model training completed successfully!")
                            
                            # Store training history
                            st.session_state.training_history = history
                            
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")
    
    with tab3:
        st.subheader("📈 Model Analytics")
        
        if 'training_history' in st.session_state:
            history = st.session_state.training_history
            
            # Training metrics plots
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy plot
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
                ax.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
                ax.set_title('Model Accuracy')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Accuracy')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                # Loss plot
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(history.history['loss'], label='Training Loss', linewidth=2)
                ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
                ax.set_title('Model Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Final metrics
            final_train_acc = history.history['accuracy'][-1]
            final_val_acc = history.history['val_accuracy'][-1]
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            
            st.markdown("### 📊 Final Training Metrics")
            
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Training Accuracy", f"{final_train_acc:.3f}")
            with metric_cols[1]:
                st.metric("Validation Accuracy", f"{final_val_acc:.3f}")
            with metric_cols[2]:
                st.metric("Training Loss", f"{final_train_loss:.3f}")
            with metric_cols[3]:
                st.metric("Validation Loss", f"{final_val_loss:.3f}")
                
        else:
            st.info("No training history available. Train a model to see analytics.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p><strong>Enhanced Diabetic Tongue Detection System</strong></p>
        <p>Built with ❤️ using Streamlit • Powered by TensorFlow/Keras</p>
        <p><em>For educational and research purposes only</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()