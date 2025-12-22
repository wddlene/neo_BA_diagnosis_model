from flask import Flask, request, jsonify, render_template
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import json

app = Flask(__name__)

# Define the neural network model structure
class NeonatalTransferLearningModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout_rate=0.4):
        super(NeonatalTransferLearningModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Build neural network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers).to(self.device)
    
    def forward(self, x):
        return self.model(x)
    
    def predict_proba(self, x):
        """Predict probabilities"""
        self.eval()
        with torch.no_grad():
            if isinstance(x, pd.DataFrame):
                x_tensor = torch.tensor(x.values, dtype=torch.float32).to(self.device)
            elif isinstance(x, np.ndarray):
                x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            else:
                # Assume it's already a tensor
                x_tensor = x.to(self.device)
            
            predictions = self.model(x_tensor)
            probs = predictions.cpu().numpy()
            
            # Convert to binary classification probability format
            prob_array = np.zeros((len(probs), 2))
            prob_array[:, 1] = probs.flatten()
            prob_array[:, 0] = 1 - probs.flatten()
            
            return prob_array
    
    def predict(self, x, threshold=0.5):
        """Predict classes"""
        proba = self.predict_proba(x)
        return (proba[:, 1] >= threshold).astype(int)

# Load model, scaler, and feature information
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load feature information
try:
    feature_info = joblib.load('feature_info.pkl')
    feature_names = feature_info['features']
    input_dim = feature_info['feature_count']
    print(f"✅ Loaded feature information: {feature_names}")
except Exception as e:
    print(f"❌ Error loading feature info: {str(e)}")
    # Default features for neonatal BA
    feature_names = ['GB_length', 'Abnormal_GEI', 'GGT', 'DBIL', 'MMP7']
    input_dim = 5

# Create model instance
model = NeonatalTransferLearningModel(input_dim)

# Load the state dictionary
try:
    state_dict = torch.load('neonatal_transfer_model.pth', map_location=device)
    model.load_state_dict(state_dict)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    # Try alternative path
    try:
        state_dict = torch.load('best_neonatal_transfer_model.pth', map_location=device)
        model.load_state_dict(state_dict)
        print("✅ Loaded best model instead")
    except:
        print("⚠️ Could not load model, using untrained model")

model.eval()

# Load scaler
try:
    scaler = joblib.load('neonatal_scaler.pkl')
    print("✅ Scaler loaded successfully")
except Exception as e:
    print(f"❌ Error loading scaler: {str(e)}")
    # Create backup scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Fit with dummy data
    dummy_data = np.zeros((1, input_dim))
    scaler.fit(dummy_data)

# Load performance metrics if available
performance_metrics = None
try:
    performance_metrics = joblib.load('performance_metrics.pkl')
    print("✅ Performance metrics loaded")
except:
    print("⚠️ Performance metrics not available")

# Feature descriptions for the UI with updated medical reference ranges
feature_descriptions = {
    'GB_length': {
        'name': 'Gallbladder Length',
        'unit': 'mm',
        'description': 'Ultrasonographic measurement of gallbladder length in neonates',
        'normal_range': '15-34 mm (Neonatal reference range)',
        'clinical_note': 'Values <15 mm may suggest gallbladder hypoplasia',
        'icon': 'fa-ultrasound',
        'min': 0,
        'max': 100
    },
    'Abnormal_GEI': {
        'name': 'Abnormal Gallbladder Emptying Index',
        'unit': 'Binary',
        'description': 'GEI = [(fasting volume − postprandial volume)/fasting volume × 100%]. Abnormal if <30%',
        'normal_range': 'GEI ≥30%: Normal emptying; GEI <30%: Abnormal emptying',
        'clinical_note': 'Reduced gallbladder emptying (<30%) is associated with biliary atresia',
        'icon': 'fa-wave-square',
        'options': {
            '0': 'Normal (GEI ≥30%)',
            '1': 'Abnormal (GEI <30%)'
        }
    },
    'GGT': {
        'name': 'Gamma-Glutamyl Transferase',
        'unit': 'U/L',
        'description': 'Liver enzyme indicating cholestasis and hepatobiliary dysfunction',
        'normal_range': '8-219 U/L (Neonatal reference range)',
        'clinical_note': 'Elevated levels (>219 U/L) suggest cholestatic liver disease',
        'icon': 'fa-flask',
        'min': 0,
        'max': 5000
    },
    'DBIL': {
        'name': 'Direct Bilirubin',
        'unit': 'μmol/L',
        'description': 'Conjugated bilirubin level indicating hepatobiliary obstruction',
        'normal_range': '0-8.6 μmol/L (Neonatal reference range)',
        'clinical_note': 'Values >8.6 μmol/L indicate conjugated hyperbilirubinemia',
        'icon': 'fa-tint',
        'min': 0,
        'max': 1000
    },
    'MMP7': {
        'name': 'Matrix Metalloproteinase-7',
        'unit': 'ng/mL',
        'description': 'Serum biomarker for biliary atresia with high diagnostic accuracy',
        'normal_range': 'Variable, but >22 ng/mL suggests biliary atresia',
        'clinical_note': 'MMP7 >22 ng/mL has high sensitivity and specificity for BA diagnosis',
        'icon': 'fa-dna',
        'min': 0,
        'max': 500
    }
}

@app.route('/')
def home():
    # Generate feature cards HTML
    feature_cards_html = ''
    for feat in feature_names:
        if feat in feature_descriptions:
            feature_cards_html += f'''
            <div class="col-md-4">
                <div class="feature-card">
                    <div class="feature-icon">
                        <i class="fas {feature_descriptions[feat]['icon']}"></i>
                    </div>
                    <h5 class="feature-name">{feature_descriptions[feat]['name']}</h5>
                    <p class="feature-description">{feature_descriptions[feat]['description']}</p>
                    <div class="feature-range">
                        <i class="fas fa-info-circle me-1"></i>
                        {feature_descriptions[feat]['normal_range']}
                    </div>
                </div>
            </div>
            '''
    
    # Generate performance metrics HTML
    performance_html = ''
    if performance_metrics:
        test_metrics = performance_metrics.get("test_metrics", {})
        auc = test_metrics.get("auc", "N/A")
        auc_ci = test_metrics.get("auc_ci", ("N/A", "N/A"))
        recall = test_metrics.get("recall", "N/A")
        specificity = test_metrics.get("specificity", "N/A")
        
        # Format values safely
        if isinstance(auc, (int, float)):
            auc_display = f"{auc:.3f}"
        else:
            auc_display = str(auc)
        
        if isinstance(auc_ci[0], (int, float)):
            ci_lower = f"{auc_ci[0]:.3f}"
        else:
            ci_lower = str(auc_ci[0])
        
        if isinstance(auc_ci[1], (int, float)):
            ci_upper = f"{auc_ci[1]:.3f}"
        else:
            ci_upper = str(auc_ci[1])
            
        if isinstance(recall, (int, float)):
            recall_display = f"{recall:.3f}"
        else:
            recall_display = str(recall)
            
        if isinstance(specificity, (int, float)):
            specificity_display = f"{specificity:.3f}"
        else:
            specificity_display = str(specificity)
        
        performance_html = f'''
        <div class="row mt-3">
            <div class="col-12">
                <div class="info-item">
                    <i class="fas fa-chart-line"></i>
                    <div>
                        <strong>Model Performance:</strong><br>
                        Test AUC: {auc_display} 
                        (95% CI: {ci_lower}-
                        {ci_upper}) | 
                        Sensitivity: {recall_display} | 
                        Specificity: {specificity_display}
                    </div>
                </div>
            </div>
        </div>
        '''
    
    html_template = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>NeoBA-Dx: Neonatal Biliary Atresia Diagnostic System</title>
        <!-- Bootstrap 5 CSS -->
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <!-- Font Awesome -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <!-- Google Fonts -->
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Source+Sans+Pro:wght@300;400;600;700&display=swap" rel="stylesheet">
        <!-- AOS Animation -->
        <link href="https://unpkg.com/aos@2.3.1/dist/aos.css" rel="stylesheet">
        <!-- Custom CSS -->
        <style>
            :root {{
                --primary-blue: #1a237e;
                --secondary-blue: #283593;
                --accent-blue: #3949ab;
                --light-blue: #e8eaf6;
                --success-green: #2e7d32;
                --warning-orange: #f57c00;
                --danger-red: #c62828;
                --dark-gray: #263238;
                --light-gray: #f5f7fa;
                --white: #ffffff;
                --gradient-primary: linear-gradient(135deg, #1a237e 0%, #3949ab 100%);
                --gradient-success: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%);
                --gradient-warning: linear-gradient(135deg, #f57c00 0%, #ff9800 100%);
                --shadow-light: 0 4px 20px rgba(0, 0, 0, 0.08);
                --shadow-medium: 0 8px 30px rgba(0, 0, 0, 0.12);
                --shadow-heavy: 0 12px 40px rgba(0, 0, 0, 0.15);
            }}
            
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Inter', sans-serif;
                background-color: var(--light-gray);
                color: var(--dark-gray);
                line-height: 1.6;
                min-height: 100vh;
                padding: 0;
            }}
            
            .navbar {{
                background: var(--gradient-primary);
                box-shadow: var(--shadow-medium);
                padding: 1rem 0;
                position: sticky;
                top: 0;
                z-index: 1000;
            }}
            
            .navbar-brand {{
                font-family: 'Source Sans Pro', sans-serif;
                font-weight: 700;
                font-size: 1.8rem;
                color: var(--white) !important;
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .navbar-brand i {{
                font-size: 2rem;
            }}
            
            .hero-section {{
                background: var(--gradient-primary);
                color: var(--white);
                padding: 4rem 0;
                margin-bottom: 3rem;
                position: relative;
                overflow: hidden;
            }}
            
            .hero-section::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 100" preserveAspectRatio="none"><path d="M0,0V100H1000V0C800,50,600,75,400,75S200,50,0,0Z" fill="%23ffffff" opacity="0.05"/></svg>');
                background-size: cover;
            }}
            
            .hero-title {{
                font-family: 'Source Sans Pro', sans-serif;
                font-weight: 700;
                font-size: 3rem;
                margin-bottom: 1rem;
                line-height: 1.2;
            }}
            
            .hero-subtitle {{
                font-size: 1.2rem;
                opacity: 0.9;
                max-width: 700px;
                margin: 0 auto;
            }}
            
            .container-main {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 0 20px;
            }}
            
            .card {{
                border: none;
                border-radius: 16px;
                box-shadow: var(--shadow-light);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                overflow: hidden;
                background: var(--white);
                margin-bottom: 2rem;
            }}
            
            .card:hover {{
                transform: translateY(-5px);
                box-shadow: var(--shadow-medium);
            }}
            
            .card-header {{
                background: var(--gradient-primary);
                color: var(--white);
                padding: 1.5rem;
                border-bottom: none;
                font-family: 'Source Sans Pro', sans-serif;
                font-weight: 600;
                font-size: 1.5rem;
            }}
            
            .card-body {{
                padding: 2rem;
            }}
            
            .section-title {{
                font-family: 'Source Sans Pro', sans-serif;
                font-weight: 600;
                color: var(--primary-blue);
                margin-bottom: 1.5rem;
                padding-bottom: 0.5rem;
                border-bottom: 3px solid var(--accent-blue);
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            
            .feature-card {{
                background: var(--white);
                border-radius: 12px;
                padding: 1.5rem;
                box-shadow: var(--shadow-light);
                transition: all 0.3s ease;
                height: 100%;
                border-left: 4px solid var(--accent-blue);
            }}
            
            .feature-card:hover {{
                transform: translateY(-3px);
                box-shadow: var(--shadow-medium);
            }}
            
            .feature-icon {{
                font-size: 2rem;
                color: var(--accent-blue);
                margin-bottom: 1rem;
            }}
            
            .feature-name {{
                font-weight: 600;
                color: var(--primary-blue);
                margin-bottom: 0.5rem;
            }}
            
            .feature-description {{
                color: #666;
                font-size: 0.9rem;
                margin-bottom: 0.5rem;
                min-height: 60px;
            }}
            
            .feature-range {{
                font-size: 0.85rem;
                color: var(--success-green);
                font-weight: 500;
                background: rgba(46, 125, 50, 0.1);
                padding: 0.5rem;
                border-radius: 6px;
                margin-top: 0.5rem;
            }}
            
            .form-label {{
                font-weight: 600;
                color: var(--primary-blue);
                margin-bottom: 0.5rem;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            
            .form-control, .form-select {{
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                padding: 12px 15px;
                font-size: 1rem;
                transition: all 0.3s ease;
                background: var(--white);
            }}
            
            .form-control:focus, .form-select:focus {{
                border-color: var(--accent-blue);
                box-shadow: 0 0 0 3px rgba(57, 73, 171, 0.1);
                outline: none;
            }}
            
            .input-hint {{
                font-size: 0.85rem;
                color: #666;
                margin-top: 0.25rem;
                padding: 0.5rem;
                background: rgba(0, 0, 0, 0.03);
                border-radius: 6px;
                border-left: 3px solid var(--accent-blue);
            }}
            
            .btn-primary {{
                background: var(--gradient-primary);
                border: none;
                border-radius: 10px;
                padding: 14px 28px;
                font-weight: 600;
                font-size: 1.1rem;
                transition: all 0.3s ease;
                display: inline-flex;
                align-items: center;
                gap: 10px;
            }}
            
            .btn-primary:hover {{
                transform: translateY(-2px);
                box-shadow: var(--shadow-medium);
            }}
            
            .model-info-card {{
                background: linear-gradient(135deg, #e8eaf6 0%, #f5f7fa 100%);
                border-radius: 12px;
                padding: 1.5rem;
                margin-bottom: 2rem;
            }}
            
            .info-item {{
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 0.8rem;
            }}
            
            .info-item i {{
                color: var(--accent-blue);
                width: 24px;
            }}
            
            .performance-badge {{
                background: var(--gradient-success);
                color: white;
                padding: 6px 12px;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: 500;
                display: inline-block;
            }}
            
            .footer {{
                background: var(--dark-gray);
                color: var(--white);
                padding: 3rem 0 1.5rem;
                margin-top: 3rem;
            }}
            
            .footer-logo {{
                font-family: 'Source Sans Pro', sans-serif;
                font-weight: 700;
                font-size: 1.5rem;
                margin-bottom: 1rem;
            }}
            
            .citation-box {{
                background: rgba(255, 255, 255, 0.1);
                border-radius: 8px;
                padding: 1rem;
                font-size: 0.9rem;
                line-height: 1.6;
                margin-top: 1rem;
            }}
            
            .reference-note {{
                background: linear-gradient(135deg, #e1f5fe 0%, #bbdefb 100%);
                border-radius: 10px;
                padding: 1rem;
                margin: 1rem 0;
                border-left: 4px solid var(--accent-blue);
            }}
            
            .reference-note h6 {{
                color: var(--primary-blue);
                margin-bottom: 0.5rem;
            }}
            
            @media (max-width: 768px) {{
                .hero-title {{
                    font-size: 2.2rem;
                }}
                
                .card-body {{
                    padding: 1.5rem;
                }}
            }}
            
            /* Loading animation */
            .spinner {{
                display: inline-block;
                width: 20px;
                height: 20px;
                border: 3px solid rgba(255,255,255,.3);
                border-radius: 50%;
                border-top-color: var(--white);
                animation: spin 1s ease-in-out infinite;
            }}
            
            @keyframes spin {{
                to {{ transform: rotate(360deg); }}
            }}
            
            /* Tooltip styling */
            .tooltip-inner {{
                max-width: 300px;
                padding: 0.5rem 0.75rem;
                background-color: var(--dark-gray);
                border-radius: 8px;
                font-size: 0.875rem;
            }}
            
            .bs-tooltip-top .tooltip-arrow::before {{
                border-top-color: var(--dark-gray);
            }}
            
            /* Validation styling */
            .is-invalid {{
                border-color: var(--danger-red) !important;
            }}
            
            .invalid-feedback {{
                color: var(--danger-red);
                font-size: 0.875rem;
                margin-top: 0.25rem;
            }}
            
            .clinical-alert {{
                background: linear-gradient(135deg, #fff3e0 0%, #ffccbc 100%);
                border: 1px solid #ff9800;
                border-radius: 10px;
                padding: 1rem;
                margin: 1rem 0;
            }}
        </style>
    </head>
    <body>
        <!-- Navigation -->
        <nav class="navbar">
            <div class="container-main">
                <a class="navbar-brand" href="/">
                    <i class="fas fa-baby-medical"></i>
                    <span>NeoBA-Dx</span>
                </a>
                <div class="navbar-text text-white">
                    Neonatal Biliary Atresia Diagnostic System
                </div>
            </div>
        </nav>
        
        <!-- Hero Section -->
        <section class="hero-section" data-aos="fade-up">
            <div class="container-main text-center">
                <h1 class="hero-title">Advanced Diagnostic Tool for Neonatal Biliary Atresia</h1>
                <p class="hero-subtitle">
                    A transfer learning-based predictive model utilizing ultrasonographic, biochemical, 
                    and serum MMP7 biomarkers for early detection of biliary atresia in neonates (≤28 days)
                </p>
            </div>
        </section>
        
        <!-- Main Content -->
        <main class="container-main">
            <!-- Model Information -->
            <div class="row mb-4" data-aos="fade-up" data-aos-delay="100">
                <div class="col-12">
                    <div class="model-info-card">
                        <div class="row">
                            <div class="col-md-4">
                                <div class="info-item">
                                    <i class="fas fa-brain"></i>
                                    <div>
                                        <strong>Model Type:</strong><br>
                                        Transfer Learning Neural Network
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="info-item">
                                    <i class="fas fa-dna"></i>
                                    <div>
                                        <strong>Biomarkers:</strong><br>
                                        Ultrasound + Biochemistry + MMP7
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-4">
                                <div class="info-item">
                                    <i class="fas fa-baby"></i>
                                    <div>
                                        <strong>Population:</strong><br>
                                        Neonates (≤28 days old)
                                    </div>
                                </div>
                            </div>
                        </div>
                        {performance_html}
                    </div>
                </div>
            </div>
            
            <!-- Feature Information -->
            <div class="row mb-4" data-aos="fade-up" data-aos-delay="150">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-microscope me-2"></i> Predictive Features Overview
                        </div>
                        <div class="card-body">
                            <div class="clinical-alert mb-4">
                                <h6><i class="fas fa-stethoscope me-2"></i>Neonatal Reference Ranges</h6>
                                <p class="mb-0">All reference values are specific to neonates (≤28 days old) as established in clinical literature.</p>
                            </div>
                            <div class="row g-4">
                                {feature_cards_html}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Input Form -->
            <div class="row" data-aos="fade-up" data-aos-delay="200">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <i class="fas fa-clipboard-list me-2"></i> Enter Neonatal Clinical Parameters
                        </div>
                        <div class="card-body">
                            <div class="alert alert-info d-flex align-items-center mb-4">
                                <i class="fas fa-info-circle me-3 fs-4"></i>
                                <div>
                                    <strong>Clinical Note:</strong> This model is specifically validated for neonates (≤28 days old). 
                                    Ensure all measurements are obtained from neonates within this age range.
                                </div>
                            </div>
                            
                            <form id="diagnosisForm" action="/predict" method="post" novalidate>
                                <!-- Ultrasound Parameters -->
                                <div class="mb-5">
                                    <h4 class="section-title">
                                        <i class="fas fa-ultrasound"></i>
                                        Ultrasonographic Parameters
                                    </h4>
                                    <div class="reference-note">
                                        <h6><i class="fas fa-book-medical me-2"></i>Ultrasound Reference</h6>
                                        <p class="mb-0">Normal neonatal gallbladder length: 15-34 mm. Gallbladder emptying index (GEI) = [(fasting volume − postprandial volume)/fasting volume × 100%]. Abnormal GEI is predefined as <30%.</p>
                                    </div>
                                    <div class="row g-3">
                                        <div class="col-md-6">
                                            <label for="GB_length" class="form-label">
                                                <i class="fas fa-ruler-vertical"></i>
                                                Gallbladder Length (mm)
                                            </label>
                                            <input type="number" class="form-control" id="GB_length" name="GB_length" 
                                                   step="0.1" min="0" max="100" required
                                                   placeholder="Enter gallbladder length in mm">
                                            <div class="input-hint">
                                                <strong>Normal neonatal range:</strong> 15-34 mm<br>
                                                <strong>Clinical significance:</strong> Values <15 mm may suggest gallbladder hypoplasia
                                            </div>
                                        </div>
                                        <div class="col-md-6">
                                            <label for="Abnormal_GEI" class="form-label">
                                                <i class="fas fa-wave-square"></i>
                                                Gallbladder Emptying Index (GEI)
                                            </label>
                                            <select class="form-select" id="Abnormal_GEI" name="Abnormal_GEI" required>
                                                <option value="" selected disabled>Select GEI status</option>
                                                <option value="0">Normal (GEI ≥30%)</option>
                                                <option value="1">Abnormal (GEI <30%)</option>
                                            </select>
                                            <div class="input-hint">
                                                <strong>Definition:</strong> GEI = [(fasting volume − postprandial volume)/fasting volume × 100%]<br>
                                                <strong>Abnormal:</strong> GEI <30% is associated with biliary atresia
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <!-- Biochemical Parameters -->
                                <div class="mb-5">
                                    <h4 class="section-title">
                                        <i class="fas fa-flask"></i>
                                        Biochemical Parameters
                                    </h4>
                                    <div class="reference-note">
                                        <h6><i class="fas fa-book-medical me-2"></i>Biochemical Reference</h6>
                                        <p class="mb-0">Normal neonatal ranges: GGT: 8-219 U/L; Direct bilirubin: 0-8.6 μmol/L. Elevated values suggest cholestatic liver disease.</p>
                                    </div>
                                    <div class="row g-3">
                                        <div class="col-md-6">
                                            <label for="GGT" class="form-label">
                                                <i class="fas fa-vial"></i>
                                                Gamma-Glutamyl Transferase (U/L)
                                            </label>
                                            <input type="number" class="form-control" id="GGT" name="GGT" 
                                                   step="0.1" min="0" max="5000" required
                                                   placeholder="Enter GGT level in U/L">
                                            <div class="input-hint">
                                                <strong>Normal neonatal range:</strong> 8-219 U/L<br>
                                                <strong>Clinical significance:</strong> Values >219 U/L suggest cholestatic liver disease
                                            </div>
                                        </div>
                                        <div class="col-md-6">
                                            <label for="DBIL" class="form-label">
                                                <i class="fas fa-tint"></i>
                                                Direct Bilirubin (μmol/L)
                                            </label>
                                            <input type="number" class="form-control" id="DBIL" name="DBIL" 
                                                   step="0.1" min="0" max="1000" required
                                                   placeholder="Enter direct bilirubin level">
                                            <div class="input-hint">
                                                <strong>Normal neonatal range:</strong> 0-8.6 μmol/L<br>
                                                <strong>Clinical significance:</strong> Values >8.6 μmol/L indicate conjugated hyperbilirubinemia
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <!-- Serum Biomarker -->
                                <div class="mb-5">
                                    <h4 class="section-title">
                                        <i class="fas fa-dna"></i>
                                        Serum Biomarker
                                    </h4>
                                    <div class="reference-note">
                                        <h6><i class="fas fa-book-medical me-2"></i>Biomarker Reference</h6>
                                        <p class="mb-0">Serum MMP7 >22 ng/mL has high sensitivity and specificity for biliary atresia diagnosis in neonates.</p>
                                    </div>
                                    <div class="row g-3">
                                        <div class="col-md-6">
                                            <label for="MMP7" class="form-label">
                                                <i class="fas fa-microscope"></i>
                                                Matrix Metalloproteinase-7 (ng/mL)
                                            </label>
                                            <input type="number" class="form-control" id="MMP7" name="MMP7" 
                                                   step="0.01" min="0" max="500" required
                                                   placeholder="Enter MMP7 level in ng/mL">
                                            <div class="input-hint">
                                                <strong>Diagnostic threshold:</strong> >22 ng/mL suggests biliary atresia<br>
                                                <strong>Clinical significance:</strong> High diagnostic accuracy for biliary atresia
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <!-- Submit Button -->
                                <div class="text-center mt-4">
                                    <button type="submit" class="btn btn-primary btn-lg" id="submitBtn">
                                        <i class="fas fa-stethoscope me-2"></i>
                                        Calculate BA Probability
                                    </button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Information Footer -->
            <div class="row mt-5" data-aos="fade-up" data-aos-delay="250">
                <div class="col-12">
                    <div class="card">
                        <div class="card-body">
                            <h5 class="section-title mb-3">
                                <i class="fas fa-file-medical-alt"></i>
                                Clinical Implementation Guidance
                            </h5>
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="info-item mb-3">
                                        <i class="fas fa-check-circle text-success"></i>
                                        <div>
                                            <strong>Intended Use:</strong> Auxiliary diagnostic tool for neonatal biliary atresia
                                        </div>
                                    </div>
                                    <div class="info-item mb-3">
                                        <i class="fas fa-exclamation-triangle text-warning"></i>
                                        <div>
                                            <strong>Clinical Validation:</strong> Requires prospective validation in clinical settings
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="info-item mb-3">
                                        <i class="fas fa-user-md text-primary"></i>
                                        <div>
                                            <strong>Decision Support:</strong> Should complement clinical judgment and other diagnostic tests
                                        </div>
                                    </div>
                                    <div class="info-item mb-3">
                                        <i class="fas fa-calendar-check text-info"></i>
                                        <div>
                                            <strong>Follow-up:</strong> Regular monitoring of model performance in clinical practice
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </main>
        
        <!-- Footer -->
        <footer class="footer">
            <div class="container-main">
                <div class="row">
                    <div class="col-md-6">
                        <div class="footer-logo">
                            <i class="fas fa-baby-medical me-2"></i>NeoBA-Dx
                        </div>
                        <p class="mb-3">
                            Neonatal Biliary Atresia Diagnostic System<br>
                            Version 2.1 | Transfer Learning Model
                        </p>
                        <p class="text-muted">
                            © 2025 Research & Development. For clinical research purposes only.
                        </p>
                    </div>
                    <div class="col-md-6">
                        <h6 class="mb-3">Recommended Citation</h6>
                        <div class="citation-box">
                            NeoBA-Dx: A Transfer Learning Model for Neonatal Biliary Atresia Diagnosis 
                            Utilizing Multimodal Biomarkers. Digital Diagnostic Tool. Version 2.1. 2025.
                        </div>
                    </div>
                </div>
                <div class="row mt-4 pt-3 border-top border-secondary">
                    <div class="col-12 text-center">
                        <p class="text-muted mb-0">
                            <small>
                                This tool is intended for research use and clinical decision support only. 
                                Not for standalone diagnostic purposes. Consult with pediatric specialists for clinical decisions.
                            </small>
                        </p>
                    </div>
                </div>
            </div>
        </footer>
        
        <!-- Bootstrap JS -->
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <!-- AOS Animation -->
        <script src="https://unpkg.com/aos@2.3.1/dist/aos.js"></script>
        <!-- jQuery -->
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <!-- Custom JS -->
        <script>
            // Initialize AOS animations
            AOS.init({{
                duration: 800,
                once: true,
                offset: 100
            }});
            
            // Form validation and submission handling
            $(document).ready(function() {{
                // Form validation
                $('#diagnosisForm').on('submit', function(e) {{
                    // Basic validation
                    let isValid = true;
                    $(this).find('input[required], select[required]').each(function() {{
                        if (!$(this).val()) {{
                            $(this).addClass('is-invalid');
                            isValid = false;
                        }} else {{
                            $(this).removeClass('is-invalid');
                        }}
                    }});
                    
                    if (!isValid) {{
                        e.preventDefault();
                        // Show validation error
                        alert('Please fill in all required fields before submitting.');
                        return false;
                    }}
                    
                    // Show loading state
                    $('#submitBtn').prop('disabled', true);
                    $('#submitBtn').html('<span class="spinner"></span> Processing...');
                }});
                
                // Real-time validation
                $('input, select').on('input change', function() {{
                    if ($(this).val()) {{
                        $(this).removeClass('is-invalid');
                    }}
                }});
                
                // Set up tooltips
                var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
                var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {{
                    return new bootstrap.Tooltip(tooltipTriggerEl);
                }});
                
                // Example data for testing (remove in production)
                $('#GB_length').val('18.5');
                $('#Abnormal_GEI').val('1');
                $('#GGT').val('285.0');
                $('#DBIL').val('95.0');
                $('#MMP7').val('14.2');
            }});
        </script>
    </body>
    </html>
    '''
    
    return html_template.format(feature_cards_html=feature_cards_html, performance_html=performance_html)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = [
            float(request.form['GB_length']),
            float(request.form['Abnormal_GEI']),
            float(request.form['GGT']),
            float(request.form['DBIL']),
            float(request.form['MMP7'])
        ]
        
        # Convert to DataFrame with correct feature names
        input_df = pd.DataFrame([data], columns=feature_names)
        
        # Scale the data
        scaled_data = scaler.transform(input_df)
        
        # Convert to Tensor and make prediction
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy()[0][0]
        
        # Three-tier risk stratification based on BA probability
        ba_probability = prediction * 100  # Convert to percentage
        
        # Determine risk level
        if ba_probability < 30:
            risk_level = "Low Risk"
            risk_description = "BA probability low (<30%)"
            color_class = "result-low"
            icon = "fa-check-circle"
            alert_type = "success"
            gradient_color = "linear-gradient(135deg, #2e7d32 0%, #4caf50 100%)"
            border_color = "#2e7d32"
        elif ba_probability <= 70:
            risk_level = "Medium Risk"
            risk_description = "BA probability medium (30%-70%)"
            color_class = "result-medium"
            icon = "fa-exclamation-circle"
            alert_type = "warning"
            gradient_color = "linear-gradient(135deg, #f57c00 0%, #ff9800 100%)"
            border_color = "#f57c00"
        else:
            risk_level = "High Risk"
            risk_description = "BA probability high (>70%)"
            color_class = "result-high"
            icon = "fa-exclamation-triangle"
            alert_type = "danger"
            gradient_color = "linear-gradient(135deg, #c62828 0%, #ef5350 100%)"
            border_color = "#c62828"
        
        # Calculate confidence
        confidence_percent = round(min(ba_probability, 100 - ba_probability), 1)
        
        # Get input values for display
        gb_length = float(request.form['GB_length'])
        abnormal_gei = int(request.form['Abnormal_GEI'])
        ggt = float(request.form['GGT'])
        dbil = float(request.form['DBIL'])
        mmp7 = float(request.form['MMP7'])
        
        # Generate clinical interpretation based on risk level
        if risk_level == "Low Risk":
            interpretation = f"""
            <strong>Low Risk - BA probability low ({ba_probability:.1f}%)</strong> 
            Clinical parameters are largely within normal ranges. Gallbladder length ({gb_length} mm) is {'within normal range (15-34 mm)' if 15 <= gb_length <= 34 else 'outside normal range'}. 
            Gallbladder emptying is {'normal (GEI ≥30%)' if abnormal_gei == 0 else 'abnormal'}. 
            GGT ({ggt} U/L) is {'within normal range (8-219 U/L)' if 8 <= ggt <= 219 else 'outside normal range'}. 
            Direct bilirubin ({dbil} μmol/L) is {'within normal range (0-8.6 μmol/L)' if dbil <= 8.6 else 'elevated'}. 
            MMP7 ({mmp7} ng/mL) is {'below diagnostic threshold for BA' if mmp7 <= 22 else 'elevated'}.
            Consider alternative etiologies for neonatal cholestasis, including infectious, metabolic, or genetic causes.
            """
        elif risk_level == "Medium Risk":
            interpretation = f"""
            <strong>Medium Risk - BA probability moderate ({ba_probability:.1f}%)</strong> 
            Clinical parameters show mixed patterns. Gallbladder length ({gb_length} mm) is {'within normal range' if 15 <= gb_length <= 34 else 'abnormal'}. 
            Gallbladder emptying is {'normal' if abnormal_gei == 0 else 'abnormal'}. 
            GGT ({ggt} U/L) is {'within normal range' if 8 <= ggt <= 219 else 'elevated'}. 
            Direct bilirubin ({dbil} μmol/L) is {'within normal range' if dbil <= 8.6 else 'elevated'}. 
            MMP7 ({mmp7} ng/mL) is {'below diagnostic threshold' if mmp7 <= 22 else 'elevated'}.
            Pediatric gastroenterology consultation is recommended for further evaluation and additional diagnostic testing.
            """
        else:  # High Risk
            interpretation = f"""
            <strong>High Risk - BA probability high ({ba_probability:.1f}%)</strong> 
            Multiple clinical parameters are abnormal and suggest high probability of BA. Gallbladder length ({gb_length} mm) is {'<15 mm, suggesting gallbladder hypoplasia' if gb_length < 15 else 'abnormal'}. 
            Gallbladder emptying is {'abnormal (GEI <30%)' if abnormal_gei == 1 else 'normal'}. 
            GGT ({ggt} U/L) is {'significantly elevated' if ggt > 219 else 'elevated'}. 
            Direct bilirubin ({dbil} μmol/L) is {'significantly elevated' if dbil > 8.6 else 'elevated'}. 
            MMP7 ({mmp7} ng/mL) is {'above diagnostic threshold, strongly suggesting BA' if mmp7 > 22 else 'elevated'}.
            Immediate referral to pediatric gastroenterology is recommended for confirmatory diagnostic procedures including intraoperative cholangiography.
            """
        
        # Get performance metrics for display
        auc_value = "N/A"
        auc_ci = ("N/A", "N/A")
        if performance_metrics:
            test_metrics = performance_metrics.get('test_metrics', {})
            auc_value = test_metrics.get('auc', 'N/A')
            auc_ci = test_metrics.get('auc_ci', ("N/A", "N/A"))
        
        # Format values for display
        if isinstance(auc_value, (int, float)):
            auc_display = f"{auc_value:.3f}"
        else:
            auc_display = auc_value
            
        if isinstance(auc_ci[0], (int, float)) and isinstance(auc_ci[1], (int, float)):
            auc_ci_display = (f"{auc_ci[0]:.3f}", f"{auc_ci[1]:.3f}")
        else:
            auc_ci_display = auc_ci
        
        # Return result page
        return f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Diagnostic Result - NeoBA-Dx</title>
            <!-- Bootstrap 5 CSS -->
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <!-- Font Awesome -->
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
            <!-- Google Fonts -->
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Source+Sans+Pro:wght@300;400;600;700&display=swap" rel="stylesheet">
            <!-- AOS Animation -->
            <link href="https://unpkg.com/aos@2.3.1/dist/aos.css" rel="stylesheet">
            <style>
                :root {{
                    --primary-blue: #1a237e;
                    --secondary-blue: #283593;
                    --accent-blue: #3949ab;
                    --light-blue: #e8eaf6;
                    --success-green: #2e7d32;
                    --warning-orange: #f57c00;
                    --danger-red: #c62828;
                    --dark-gray: #263238;
                    --light-gray: #f5f7fa;
                    --white: #ffffff;
                    --gradient-primary: linear-gradient(135deg, #1a237e 0%, #3949ab 100%);
                    --gradient-success: linear-gradient(135deg, #2e7d32 0%, #4caf50 100%);
                    --gradient-warning: linear-gradient(135deg, #f57c00 0%, #ff9800 100%);
                    --gradient-danger: linear-gradient(135deg, #c62828 0%, #ef5350 100%);
                    --shadow-light: 0 4px 20px rgba(0, 0, 0, 0.08);
                    --shadow-medium: 0 8px 30px rgba(0, 0, 0, 0.12);
                    --shadow-heavy: 0 12px 40px rgba(0, 0, 0, 0.15);
                }}
                
                body {{
                    font-family: 'Inter', sans-serif;
                    background-color: var(--light-gray);
                    color: var(--dark-gray);
                    line-height: 1.6;
                    min-height: 100vh;
                }}
                
                .navbar {{
                    background: var(--gradient-primary);
                    box-shadow: var(--shadow-medium);
                    padding: 1rem 0;
                }}
                
                .navbar-brand {{
                    font-family: 'Source Sans Pro', sans-serif;
                    font-weight: 700;
                    font-size: 1.8rem;
                    color: var(--white) !important;
                }}
                
                .container-main {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 2rem 20px;
                }}
                
                .result-header {{
                    background: var(--gradient-primary);
                    color: var(--white);
                    padding: 3rem 0;
                    margin-bottom: 2rem;
                    border-radius: 16px 16px 0 0;
                    position: relative;
                    overflow: hidden;
                }}
                
                .result-header::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 100" preserveAspectRatio="none"><path d="M0,0V100H1000V0C800,50,600,75,400,75S200,50,0,0Z" fill="%23ffffff" opacity="0.05"/></svg>');
                    background-size: cover;
                }}
                
                .result-card {{
                    background: var(--white);
                    border-radius: 16px;
                    box-shadow: var(--shadow-heavy);
                    overflow: hidden;
                    margin-bottom: 2rem;
                    border: 1px solid rgba(0, 0, 0, 0.05);
                }}
                
                .result-badge {{
                    font-size: 1.8rem;
                    padding: 1.5rem 3rem;
                    border-radius: 50px;
                    text-align: center;
                    margin: 2rem auto;
                    max-width: 600px;
                    font-weight: 700;
                    box-shadow: var(--shadow-medium);
                    position: relative;
                    overflow: hidden;
                    border: none;
                }}
                
                .result-badge::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: {gradient_color};
                    z-index: 1;
                }}
                
                .result-badge-content {{
                    position: relative;
                    z-index: 2;
                }}
                
                .result-low {{
                    color: var(--white);
                }}
                
                .result-medium {{
                    color: var(--white);
                }}
                
                .result-high {{
                    color: var(--white);
                }}
                
                .confidence-meter-container {{
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    border-radius: 16px;
                    padding: 2rem;
                    margin: 2rem 0;
                    border: 1px solid rgba(0, 0, 0, 0.05);
                }}
                
                .confidence-meter {{
                    height: 30px;
                    background-color: #e9ecef;
                    border-radius: 15px;
                    overflow: hidden;
                    margin: 1.5rem 0;
                    position: relative;
                    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
                }}
                
                .confidence-fill {{
                    height: 100%;
                    border-radius: 15px;
                    background: {gradient_color};
                    width: 0%;
                    transition: width 1.5s ease-in-out;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
                }}
                
                .confidence-text {{
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    color: var(--white);
                    font-weight: bold;
                    font-size: 1.1rem;
                    text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
                }}
                
                .feature-value {{
                    font-weight: 600;
                    color: var(--primary-blue);
                }}
                
                .parameter-analysis {{
                    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin: 1.5rem 0;
                    border-left: 4px solid var(--accent-blue);
                }}
                
                .parameter-item {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 0.75rem 0;
                    border-bottom: 1px solid rgba(0, 0, 0, 0.08);
                }}
                
                .parameter-item:last-child {{
                    border-bottom: none;
                }}
                
                .parameter-status {{
                    padding: 0.25rem 0.75rem;
                    border-radius: 20px;
                    font-size: 0.85rem;
                    font-weight: 500;
                    min-width: 120px;
                    text-align: center;
                }}
                
                .status-normal {{
                    background-color: rgba(46, 125, 50, 0.15);
                    color: var(--success-green);
                    border: 1px solid rgba(46, 125, 50, 0.3);
                }}
                
                .status-abnormal {{
                    background-color: rgba(198, 40, 40, 0.15);
                    color: var(--danger-red);
                    border: 1px solid rgba(198, 40, 40, 0.3);
                }}
                
                .status-borderline {{
                    background-color: rgba(245, 124, 0, 0.15);
                    color: var(--warning-orange);
                    border: 1px solid rgba(245, 124, 0, 0.3);
                }}
                
                .reference-box {{
                    background-color: #f8f9fa;
                    border-left: 4px solid var(--accent-blue);
                    padding: 1.5rem;
                    margin: 1.5rem 0;
                    border-radius: 0 8px 8px 0;
                    box-shadow: var(--shadow-light);
                }}
                
                .btn-back {{
                    background: var(--gradient-primary);
                    border: none;
                    border-radius: 10px;
                    padding: 12px 24px;
                    font-weight: 600;
                    color: var(--white);
                    transition: all 0.3s ease;
                    box-shadow: var(--shadow-light);
                }}
                
                .btn-back:hover {{
                    transform: translateY(-2px);
                    box-shadow: var(--shadow-medium);
                    color: var(--white);
                }}
                
                .disclaimer {{
                    background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
                    border-left: 4px solid #ffb300;
                    padding: 1.5rem;
                    margin: 2rem 0;
                    border-radius: 0 8px 8px 0;
                }}
                
                .risk-level-indicator {{
                    display: flex;
                    justify-content: space-between;
                    margin: 2rem 0;
                    position: relative;
                    height: 60px;
                }}
                
                .risk-level-indicator::before {{
                    content: '';
                    position: absolute;
                    top: 50%;
                    left: 0;
                    right: 0;
                    height: 6px;
                    background: linear-gradient(90deg, var(--success-green) 0%, var(--warning-orange) 50%, var(--danger-red) 100%);
                    transform: translateY(-50%);
                    z-index: 1;
                    border-radius: 3px;
                }}
                
                .risk-marker {{
                    width: 30px;
                    height: 30px;
                    border-radius: 50%;
                    background: var(--white);
                    border: 3px solid;
                    position: relative;
                    z-index: 2;
                    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
                }}
                
                .risk-label {{
                    position: absolute;
                    top: 40px;
                    transform: translateX(-50%);
                    font-size: 0.9rem;
                    font-weight: 600;
                    text-align: center;
                    width: 100px;
                }}
                
                .risk-label small {{
                    display: block;
                    font-weight: normal;
                    font-size: 0.8rem;
                    opacity: 0.8;
                }}
                
                .probability-display {{
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin: 1rem 0;
                    text-align: center;
                    border: 2px solid {border_color};
                }}
                
                .probability-number {{
                    font-size: 3rem;
                    font-weight: 700;
                    color: {border_color};
                    margin-bottom: 0.5rem;
                }}
                
                .probability-label {{
                    font-size: 1.2rem;
                    color: var(--dark-gray);
                    font-weight: 500;
                }}
                
                .section-title {{
                    font-family: 'Source Sans Pro', sans-serif;
                    font-weight: 600;
                    color: var(--primary-blue);
                    margin-bottom: 1.5rem;
                    padding-bottom: 0.5rem;
                    border-bottom: 3px solid var(--accent-blue);
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}
                
                .info-item {{
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin-bottom: 0.8rem;
                }}
                
                .info-item i {{
                    color: var(--accent-blue);
                    width: 24px;
                }}
                
                .model-performance {{
                    background: linear-gradient(135deg, #e8eaf6 0%, #f5f7fa 100%);
                    border-radius: 12px;
                    padding: 1.5rem;
                    margin-bottom: 2rem;
                }}
            </style>
        </head>
        <body>
            <!-- Navigation -->
            <nav class="navbar">
                <div class="container-fluid" style="max-width: 1200px; margin: 0 auto;">
                    <a class="navbar-brand" href="/">
                        <i class="fas fa-baby-medical me-2"></i>
                        NeoBA-Dx: Diagnostic Result
                    </a>
                </div>
            </nav>
            
            <!-- Main Content -->
            <div class="container-main">
                <!-- Result Header -->
                <div class="result-header text-center">
                    <h1 class="display-5 fw-bold mb-3">Diagnostic Analysis Complete</h1>
                    <p class="lead">Neonatal Biliary Atresia Risk Assessment</p>
                </div>
                
                <!-- Main Result Card -->
                <div class="result-card">
                    <div class="p-4 p-md-5">
                        <!-- Diagnosis Result -->
                        <div class="text-center mb-5">
                            <div class="result-badge {color_class}">
                                <div class="result-badge-content">
                                    <i class="fas {icon} me-3"></i>
                                    {risk_level}: {risk_description}
                                </div>
                            </div>
                            
                            <!-- Probability Display -->
                            <div class="row justify-content-center">
                                <div class="col-md-6">
                                    <div class="probability-display" data-aos="fade-up">
                                        <div class="probability-number">{ba_probability:.1f}%</div>
                                        <div class="probability-label">BA Probability</div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Risk Level Indicator -->
                            <div class="row justify-content-center">
                                <div class="col-md-10">
                                    <div class="risk-level-indicator" data-aos="fade-up" data-aos-delay="100">
                                        <div class="risk-marker" style="border-color: var(--success-green); left: 15%;"></div>
                                        <div class="risk-marker" style="border-color: var(--warning-orange); left: 50%;"></div>
                                        <div class="risk-marker" style="border-color: var(--danger-red); left: 85%;"></div>
                                        <div class="risk-label" style="left: 15%;">Low Risk<br><small><30%</small></div>
                                        <div class="risk-label" style="left: 50%;">Medium Risk<br><small>30-70%</small></div>
                                        <div class="risk-label" style="left: 85%;">High Risk<br><small>>70%</small></div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Confidence Meter -->
                            <div class="confidence-meter-container" data-aos="fade-up" data-aos-delay="200">
                                <h5 class="text-center mb-4">
                                    <i class="fas fa-chart-line me-2"></i>
                                    BA Probability Visualization
                                </h5>
                                <div class="row justify-content-center">
                                    <div class="col-md-10">
                                        <div class="confidence-meter">
                                            <div class="confidence-fill" id="confidenceFill"></div>
                                            <div class="confidence-text" id="confidenceText">0%</div>
                                        </div>
                                        <div class="d-flex justify-content-between mt-2">
                                            <small class="text-muted">BA Probability: 0%</small>
                                            <small class="text-muted">BA Probability: 100%</small>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Probability Details -->
                        <div class="model-performance" data-aos="fade-up" data-aos-delay="300">
                            <h5 class="mb-3">
                                <i class="fas fa-chart-bar me-2"></i>
                                Probability Analysis
                            </h5>
                            <div class="row">
                                <div class="col-md-6">
                                    <div class="info-item">
                                        <i class="fas fa-percentage"></i>
                                        <div>
                                            <strong>BA Probability Score:</strong><br>
                                            <span class="feature-value">{prediction:.4f}</span>
                                        </div>
                                    </div>
                                    <div class="info-item">
                                        <i class="fas fa-chart-pie"></i>
                                        <div>
                                            <strong>BA Probability (%):</strong><br>
                                            <span class="feature-value">{ba_probability:.1f}%</span>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <div class="info-item">
                                        <i class="fas fa-shield-alt"></i>
                                        <div>
                                            <strong>Risk Level:</strong><br>
                                            <span class="feature-value">{risk_level}</span>
                                        </div>
                                    </div>
                                    <div class="info-item">
                                        <i class="fas fa-tachometer-alt"></i>
                                        <div>
                                            <strong>Model Performance:</strong><br>
                                            <span class="feature-value">AUC = {auc_display} (95% CI: {auc_ci_display[0]}-{auc_ci_display[1]})</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Parameter Analysis -->
                        <div class="mt-5" data-aos="fade-up" data-aos-delay="400">
                            <h5 class="section-title mb-4">
                                <i class="fas fa-clipboard-check me-2"></i>
                                Parameter Analysis vs. Reference Ranges
                            </h5>
                            <div class="parameter-analysis">
                                <div class="parameter-item">
                                    <div>
                                        <i class="fas fa-ultrasound me-2 text-primary"></i>
                                        <strong>Gallbladder Length:</strong> {gb_length} mm
                                    </div>
                                    <span class="parameter-status {'status-normal' if 15 <= gb_length <= 34 else 'status-abnormal'}">
                                        {'Normal (15-34 mm)' if 15 <= gb_length <= 34 else 'Abnormal'}
                                    </span>
                                </div>
                                <div class="parameter-item">
                                    <div>
                                        <i class="fas fa-wave-square me-2 text-primary"></i>
                                        <strong>Gallbladder Emptying Index:</strong> {['Normal (GEI ≥30%)', 'Abnormal (GEI <30%)'][abnormal_gei]}
                                    </div>
                                    <span class="parameter-status {'status-normal' if abnormal_gei == 0 else 'status-abnormal'}">
                                        {['Normal', 'Abnormal'][abnormal_gei]}
                                    </span>
                                </div>
                                <div class="parameter-item">
                                    <div>
                                        <i class="fas fa-flask me-2 text-primary"></i>
                                        <strong>GGT:</strong> {ggt} U/L
                                    </div>
                                    <span class="parameter-status {'status-normal' if 8 <= ggt <= 219 else 'status-abnormal'}">
                                        {'Normal (8-219 U/L)' if 8 <= ggt <= 219 else 'Elevated'}
                                    </span>
                                </div>
                                <div class="parameter-item">
                                    <div>
                                        <i class="fas fa-tint me-2 text-primary"></i>
                                        <strong>Direct Bilirubin:</strong> {dbil} μmol/L
                                    </div>
                                    <span class="parameter-status {'status-normal' if dbil <= 8.6 else 'status-abnormal'}">
                                        {'Normal (0-8.6 μmol/L)' if dbil <= 8.6 else 'Elevated'}
                                    </span>
                                </div>
                                <div class="parameter-item">
                                    <div>
                                        <i class="fas fa-dna me-2 text-primary"></i>
                                        <strong>MMP7:</strong> {mmp7} ng/mL
                                    </div>
                                    <span class="parameter-status {'status-normal' if mmp7 <= 22 else 'status-abnormal'}">
                                        {'Normal (≤22 ng/mL)' if mmp7 <= 22 else 'Elevated (>22 ng/mL suggests BA)'}
                                    </span>
                                </div>
                            </div>
                            
                            <!-- Reference Box -->
                            <div class="reference-box">
                                <h6><i class="fas fa-book-medical me-2"></i>Neonatal Reference Ranges</h6>
                                <ul class="mb-0">
                                    <li><i class="fas fa-ultrasound me-2 text-muted"></i><strong>Gallbladder Length:</strong> 15-34 mm</li>
                                    <li><i class="fas fa-wave-square me-2 text-muted"></i><strong>Gallbladder Emptying Index (GEI):</strong> GEI = [(fasting volume − postprandial volume)/fasting volume × 100%]</li>
                                    <li><i class="fas fa-exclamation-triangle me-2 text-muted"></i><strong>Abnormal GEI:</strong> <30%</li>
                                    <li><i class="fas fa-flask me-2 text-muted"></i><strong>GGT:</strong> 8-219 U/L</li>
                                    <li><i class="fas fa-tint me-2 text-muted"></i><strong>Direct Bilirubin:</strong> 0-8.6 μmol/L</li>
                                    <li><i class="fas fa-dna me-2 text-muted"></i><strong>MMP7 diagnostic threshold for BA:</strong> >22 ng/mL</li>
                                </ul>
                            </div>
                        </div>
                        
                        <!-- Clinical Interpretation -->
                        <div class="alert alert-{alert_type} mt-5" data-aos="fade-up" data-aos-delay="500">
                            <h5 class="alert-heading">
                                <i class="fas {icon} me-2"></i>
                                Clinical Interpretation
                            </h5>
                            <div class="mt-2">
                                {interpretation}
                            </div>
                        </div>
                        
                        <!-- Recommendations -->
                        <div class="disclaimer mt-5" data-aos="fade-up" data-aos-delay="600">
                            <h5 class="mb-3">
                                <i class="fas fa-user-md me-2"></i>
                                Clinical Recommendations
                            </h5>
                            <ul class="mb-0">
                                <li class="mb-2"><i class="fas fa-check-circle me-2 text-success"></i>This AI-assisted assessment should be interpreted in conjunction with comprehensive clinical evaluation</li>
                                <li class="mb-2"><i class="fas fa-stethoscope me-2 text-primary"></i>Consultation with pediatric gastroenterology is recommended for definitive diagnosis</li>
                                <li class="mb-2"><i class="fas fa-microscope me-2 text-info"></i>Consider additional diagnostic tests (abdominal ultrasound, MRCP, hepatobiliary scintigraphy, liver biopsy) based on clinical suspicion</li>
                                <li class="mb-2"><i class="fas fa-clock me-2 text-warning"></i>If biliary atresia is confirmed, prompt referral for Kasai portoenterostomy is essential (optimal before 60 days of life)</li>
                                <li><i class="fas fa-file-medical me-2 text-secondary"></i>Document this assessment in the patient's medical record as part of the diagnostic workup</li>
                            </ul>
                        </div>
                        
                        <!-- Model Information -->
                        <div class="reference-box mt-5" data-aos="fade-up" data-aos-delay="700">
                            <h6><i class="fas fa-info-circle me-2"></i>Model Information</h6>
                            <div class="row">
                                <div class="col-md-6">
                                    <p class="mb-2"><strong><i class="fas fa-brain me-2"></i>Model:</strong> Transfer Learning Neural Network (pre-trained with Random Forest)</p>
                                    <p class="mb-2"><strong><i class="fas fa-dna me-2"></i>Biomarkers:</strong> Gallbladder length, Gallbladder emptying index, GGT, Direct bilirubin, Serum MMP7</p>
                                </div>
                                <div class="col-md-6">
                                    <p class="mb-2"><strong><i class="fas fa-baby me-2"></i>Population:</strong> Neonates ≤28 days old</p>
                                    <p class="mb-0"><strong><i class="fas fa-crosshairs me-2"></i>Purpose:</strong> Auxiliary diagnostic tool for neonatal biliary atresia risk assessment</p>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Action Buttons -->
                        <div class="d-flex justify-content-between mt-5 pt-4 border-top" data-aos="fade-up" data-aos-delay="800">
                            <a href="/" class="btn btn-back">
                                <i class="fas fa-arrow-left me-2"></i>
                                New Assessment
                            </a>
                            <div>
                                <button onclick="window.print()" class="btn btn-back me-2">
                                    <i class="fas fa-print me-2"></i>
                                    Print Report
                                </button>
                                <button onclick="downloadReport()" class="btn btn-back">
                                    <i class="fas fa-download me-2"></i>
                                    Download PDF
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Footer Note -->
                <div class="text-center mt-4 text-muted">
                    <small>
                        <i class="fas fa-shield-alt me-1"></i>NeoBA-Dx v2.1 | This tool is for clinical decision support only. 
                        Final diagnosis requires comprehensive evaluation by qualified pediatric specialists.
                        © 2025 Research & Development.
                    </small>
                </div>
            </div>
            
            <!-- Bootstrap JS -->
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
            <!-- AOS Animation -->
            <script src="https://unpkg.com/aos@2.3.1/dist/aos.js"></script>
            <!-- jQuery -->
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <!-- Custom JS -->
            <script>
                // Initialize AOS animations
                AOS.init({{
                    duration: 800,
                    once: true,
                    offset: 100
                }});
                
                // Animate confidence meter
                document.addEventListener('DOMContentLoaded', function() {{
                    const fill = document.getElementById('confidenceFill');
                    const text = document.getElementById('confidenceText');
                    const targetWidth = {ba_probability};
                    
                    let currentWidth = 0;
                    const interval = setInterval(() => {{
                        if (currentWidth >= targetWidth) {{
                            clearInterval(interval);
                            text.textContent = '{ba_probability:.1f}%';
                        }} else {{
                            currentWidth++;
                            fill.style.width = currentWidth + '%';
                            text.textContent = currentWidth.toFixed(0) + '%';
                        }}
                    }}, 15);
                }});
                
                // Download report function
                function downloadReport() {{
                    const resultData = {{
                        risk_level: '{risk_level}',
                        risk_description: '{risk_description}',
                        ba_probability: {ba_probability},
                        probability_score: {prediction},
                        parameters: {{
                            GB_length: '{request.form['GB_length']}',
                            Abnormal_GEI: '{request.form['Abnormal_GEI']}',
                            GGT: '{request.form['GGT']}',
                            DBIL: '{request.form['DBIL']}',
                            MMP7: '{request.form['MMP7']}'
                        }},
                        parameter_analysis: {{
                            gallbladder_length: {{
                                value: {gb_length},
                                normal_range: "15-34 mm",
                                status: "{'Normal' if 15 <= gb_length <= 34 else 'Abnormal'}"
                            }},
                            gei: {{
                                value: {abnormal_gei},
                                description: "{'Normal (GEI ≥30%)' if abnormal_gei == 0 else 'Abnormal (GEI <30%)'}",
                                status: "{'Normal' if abnormal_gei == 0 else 'Abnormal'}"
                            }},
                            ggt: {{
                                value: {ggt},
                                normal_range: "8-219 U/L",
                                status: "{'Normal' if 8 <= ggt <= 219 else 'Elevated'}"
                            }},
                            dbil: {{
                                value: {dbil},
                                normal_range: "0-8.6 μmol/L",
                                status: "{'Normal' if dbil <= 8.6 else 'Elevated'}"
                            }},
                            mmp7: {{
                                value: {mmp7},
                                diagnostic_threshold: ">22 ng/mL",
                                status: "{'Normal' if mmp7 <= 22 else 'Elevated'}"
                            }}
                        }},
                        timestamp: new Date().toISOString(),
                        model: 'NeoBA-Dx v2.1'
                    }};
                    
                    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(resultData, null, 2));
                    const downloadAnchor = document.createElement('a');
                    downloadAnchor.setAttribute("href", dataStr);
                    downloadAnchor.setAttribute("download", "neobadx_report_" + new Date().getTime() + ".json");
                    document.body.appendChild(downloadAnchor);
                    downloadAnchor.click();
                    downloadAnchor.remove();
                }}
            </script>
        </body>
        </html>
        '''
    
    except Exception as e:
        return f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Error - NeoBA-Dx</title>
            <!-- Bootstrap CSS -->
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body {{
                    font-family: 'Inter', sans-serif;
                    background: linear-gradient(135deg, #f5f7fa 0%, #e3e7eb 100%);
                    min-height: 100vh;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }}
                
                .error-card {{
                    max-width: 600px;
                    border-radius: 16px;
                    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
                    overflow: hidden;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="error-card">
                    <div class="card-header bg-danger text-white">
                        <h4 class="mb-0"><i class="fas fa-exclamation-triangle me-2"></i>Prediction Error</h4>
                    </div>
                    <div class="card-body text-center p-5">
                        <div class="display-1 text-danger mb-4">
                            <i class="fas fa-times-circle"></i>
                        </div>
                        <h4 class="mb-3">Error Processing Diagnostic Request</h4>
                        <div class="alert alert-danger mb-4">
                            <strong>Error Details:</strong> {str(e)}
                        </div>
                        <p class="text-muted mb-4">
                            Please verify that all input values are valid numeric entries and try again.
                        </p>
                        <a href="/" class="btn btn-primary btn-lg">
                            <i class="fas fa-arrow-left me-2"></i>Return to Input Form
                        </a>
                    </div>
                </div>
            </div>
        </body>
        </html>
        '''

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['GB_length', 'Abnormal_GEI', 'GGT', 'DBIL', 'MMP7']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing required field: {field}'
                }), 400
        
        # Extract features
        input_data = [
            float(data['GB_length']),
            float(data['Abnormal_GEI']),
            float(data['GGT']),
            float(data['DBIL']),
            float(data['MMP7'])
        ]
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data], columns=feature_names)
        
        # Scale the data
        scaled_data = scaler.transform(input_df)
        
        # Convert to Tensor and make prediction
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy()[0][0]
        
        # Calculate BA probability percentage
        ba_probability = prediction * 100
        
        # Determine risk level based on the three-tier system
        if ba_probability < 30:
            risk_level = "Low Risk"
            risk_description = "BA probability low (<30%)"
        elif ba_probability <= 70:
            risk_level = "Medium Risk"
            risk_description = "BA probability medium (30%-70%)"
        else:
            risk_level = "High Risk"
            risk_description = "BA probability high (>70%)"
        
        response = {
            'success': True,
            'risk_level': risk_level,
            'risk_description': risk_description,
            'ba_probability': float(ba_probability),
            'probability_score': float(prediction),
            'threshold': 0.5,
            'timestamp': pd.Timestamp.now().isoformat(),
            'model': {
                'name': 'NeoBA-Dx',
                'version': '2.1',
                'type': 'Transfer Learning Neural Network',
                'features': feature_names
            },
            'reference_ranges': {
                'GB_length': {'normal': '15-34 mm', 'unit': 'mm'},
                'Abnormal_GEI': {'normal': 'GEI ≥30%', 'abnormal': 'GEI <30%'},
                'GGT': {'normal': '8-219 U/L', 'unit': 'U/L'},
                'DBIL': {'normal': '0-8.6 μmol/L', 'unit': 'μmol/L'},
                'MMP7': {'diagnostic_threshold': '>22 ng/mL', 'unit': 'ng/mL'}
            },
            'risk_stratification': {
                'low': 'BA probability <30%',
                'medium': 'BA probability 30%-70%',
                'high': 'BA probability >70%'
            },
            'performance': {
                'auc': performance_metrics.get('test_metrics', {}).get('auc', 'N/A') if performance_metrics else 'N/A',
                'sensitivity': performance_metrics.get('test_metrics', {}).get('recall', 'N/A') if performance_metrics else 'N/A',
                'specificity': performance_metrics.get('test_metrics', {}).get('specificity', 'N/A') if performance_metrics else 'N/A'
            }
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': pd.Timestamp.now().isoformat()
        }), 400

@app.route('/api/model_info', methods=['GET'])
def api_model_info():
    """API endpoint to get model information"""
    info = {
        'model_name': 'NeoBA-Dx',
        'version': '2.1',
        'model_type': 'Transfer Learning Neural Network',
        'features': feature_names,
        'feature_descriptions': feature_descriptions,
        'population': 'Neonates (≤28 days old)',
        'purpose': 'Auxiliary diagnostic tool for neonatal biliary atresia risk assessment',
        'reference_ranges': {
            'GB_length': '15-34 mm',
            'Abnormal_GEI': 'GEI <30% is abnormal (GEI = [(fasting volume − postprandial volume)/fasting volume × 100%])',
            'GGT': '8-219 U/L',
            'DBIL': '0-8.6 μmol/L',
            'MMP7': '>22 ng/mL suggests biliary atresia'
        },
        'risk_stratification': {
            'low_risk': 'BA probability <30%',
            'medium_risk': 'BA probability 30%-70%',
            'high_risk': 'BA probability >70%'
        },
        'performance': performance_metrics.get('test_metrics', {}) if performance_metrics else {}
    }
    return jsonify(info)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)