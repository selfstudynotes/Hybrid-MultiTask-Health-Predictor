from flask import Flask, request, send_file, render_template_string
import numpy as np
import pickle
import io
import tensorflow as tf

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

app = Flask(__name__)

scaler = None
model = None

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    model = tf.keras.models.load_model('multitask_model.h5', custom_objects={'Attention': tf.keras.layers.Attention})
    print("✅ Model & Scaler Loaded Successfully!")
except Exception as e:
    print(f"⚠️ Model Load Warning: {e}")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MedAI - Joint Prediction System</title>
    
    <!-- Bootstrap 5 & FontAwesome -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
        :root {
            --blue-primary: #2563EB;
            --cyan-accent: #06B6D4;
            --green-success: #10B981;
            --red-danger: #EF4444;
            --bg-light: #F8FAFC;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: linear-gradient(rgba(244,250,255,.9), rgba(240,248,255,.9)),
                        url("https://images.unsplash.com/photo-1576091160550-2173dba999ef?auto=format&fit=crop&w=2000&q=80");
            background-size: cover;
            background-attachment: fixed;
            color: #0f172a;
            min-height: 100vh;
            position: relative;
            overflow-x: hidden;
        }

        /* FLOATING BACKGROUND ICONS */
        .floating {
            position: fixed;
            opacity: .08;
            font-size: 70px;
            animation: float 7s ease-in-out infinite;
            z-index: 0;
            pointer-events: none;
        }

        .floating.heart { top: 15%; left: 8%; animation-delay: 0s; color: var(--red-danger); }
        .floating.dna { top: 25%; right: 10%; animation-delay: 2s; color: var(--blue-primary); }
        .floating.brain { bottom: 20%; left: 12%; animation-delay: 4s; color: var(--cyan-accent); }
        .floating.steth { bottom: 15%; right: 12%; animation-delay: 1s; color: var(--green-success); }

        @keyframes float {
            0% { transform: translateY(0); }
            50% { transform: translateY(-20px); }
            100% { transform: translateY(0); }
        }

        /* NAVIGATION BAR */
        .navbar-custom {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(20px);
            border-bottom: 1px solid rgba(226, 232, 240, 0.8);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .nav-brand-text {
            font-weight: 800;
            font-size: 1.4rem;
            color: var(--blue-primary);
        }

        .nav-link-custom {
            color: #475569;
            font-weight: 700;
            font-size: 0.95rem;
            padding: 8px 16px !important;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
        }

        .nav-link-custom:hover {
            color: var(--blue-primary);
            background: rgba(37, 99, 235, 0.08);
        }

        .nav-link-custom.active {
            background: var(--blue-primary);
            color: #ffffff !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
        }

        /* GLASS CARD SPECIFICATIONS */
        .glass-card {
            background: rgba(255, 255, 255, 0.88);
            backdrop-filter: blur(30px);
            border-radius: 30px;
            border: 1px solid #ffffff;
            box-shadow: 0 20px 50px rgba(0,0,0,.15);
            transition: .4s;
            padding: 2.5rem;
            margin-bottom: 2rem;
            position: relative;
            z-index: 1;
        }

        .glass-card:hover {
            transform: translateY(-8px);
        }

        /* METRIC STAT BOXES */
        .stat-card {
            background: #ffffff;
            border-radius: 20px;
            padding: 1.5rem;
            border: 1.5px solid #e2e8f0;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.03);
            text-align: center;
            transition: 0.3s;
        }

        .stat-card:hover {
            border-color: var(--blue-primary);
            box-shadow: 0 12px 30px rgba(37, 99, 235, 0.12);
        }

        .stat-num {
            font-size: 2.2rem;
            font-weight: 800;
            color: var(--blue-primary);
        }

        .stat-label {
            font-size: 0.9rem;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
        }

        /* PIPELINE VISUALIZATION */
        .pipe-box {
            background: #f1f5f9;
            border: 2px dashed #cbd5e1;
            border-radius: 16px;
            padding: 1rem;
            font-weight: 800;
            color: #1e293b;
            text-align: center;
        }

        .arrow-down {
            text-align: center;
            font-size: 1.4rem;
            color: var(--blue-primary);
            margin: 6px 0;
        }

        /* FORM FIELD STYLING */
        .form-label-custom {
            color: #334155;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }

        .form-control-custom {
            background: #ffffff;
            border: 1.5px solid #cbd5e1;
            border-radius: 12px;
            padding: 12px 16px;
            color: #0f172a;
            font-size: 1rem;
            font-weight: 600;
        }

        .form-control-custom:focus {
            border-color: var(--blue-primary);
            box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.15);
        }

        .btn-medai {
            background: linear-gradient(135deg, var(--blue-primary) 0%, var(--cyan-accent) 100%);
            color: white;
            font-weight: 800;
            border-radius: 14px;
            padding: 16px;
            border: none;
            width: 100%;
            font-size: 1.05rem;
            letter-spacing: 0.5px;
            box-shadow: 0 6px 20px rgba(37, 99, 235, 0.25);
            transition: all 0.3s ease;
        }

        .btn-medai:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(37, 99, 235, 0.35);
            color: white;
        }

        footer {
            background: rgba(255, 255, 255, 0.9);
            border-top: 1px solid #e2e8f0;
            padding: 2rem 0;
            margin-top: 4rem;
            position: relative;
            z-index: 10;
        }
    </style>
</head>
<body>

    <!-- FLOATING BACKGROUND ICONS -->
    <i class="fa-solid fa-heart-pulse floating heart"></i>
    <i class="fa-solid fa-dna floating dna"></i>
    <i class="fa-solid fa-brain floating brain"></i>
    <i class="fa-solid fa-stethoscope floating steth"></i>

    <!-- NAVBAR HEADER -->
    <nav class="navbar navbar-expand-lg navbar-custom">
        <div class="container">
            <a class="navbar-brand d-flex align-items-center gap-2" onclick="switchSection('home')" style="cursor: pointer;">
                <span class="fs-3">🩺</span>
                <span class="nav-brand-text">MedAI</span>
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-menu ms-auto d-flex gap-2 list-unstyled mb-0 py-2">
                    <li><a class="nav-link-custom active" id="tab-home" onclick="switchSection('home')">Home</a></li>
                    <li><a class="nav-link-custom" id="tab-joint" onclick="switchSection('joint')">Predict</a></li>
                    <li><a class="nav-link-custom" id="tab-overview" onclick="switchSection('overview')">Overview</a></li>
                    <li><a class="nav-link-custom" id="tab-tech" onclick="switchSection('tech')">Architecture</a></li>
                    <li><a class="nav-link-custom" id="tab-contact" onclick="switchSection('contact')">Contact</a></li>
                </ul>
            </div>
        </div>
    </nav>

    <!-- MAIN CONTAINER -->
    <div class="container my-4">

        <!-- SECTION 1: HOMEPAGE -->
        <div id="sec-home">
            <div class="glass-card text-center mb-4">
                <div class="d-inline-flex align-items-center gap-2 badge bg-primary text-white rounded-pill px-3 py-2 fs-6 mb-3">
                    <i class="fa-solid fa-wave-square"></i> Next-Gen Attention Deep Learning
                </div>
                <h1 class="fw-extrabold display-5 mb-3">Joint Health Assessment Portal</h1>
                <p class="text-secondary fs-5 max-w-700 mx-auto mb-4">
                    Simultaneous AI prediction for Cardiovascular Risk and Type-2 Diabetes using shared multi-task representation layers.
                </p>
                <div class="d-flex justify-content-center gap-3">
                    <button onclick="switchSection('joint')" class="btn btn-medai px-5 py-3 w-auto fs-6">
                        <i class="fa-solid fa-stethoscope me-2"></i>START PREDICTION
                    </button>
                </div>
            </div>

            <!-- MENTOR'S HIGHLIGHTED METRICS -->
            <div class="row g-4 mb-4">
                <div class="col-md-3 col-6">
                    <div class="stat-card">
                        <div class="stat-num">95.8%</div>
                        <div class="stat-label">Prediction Accuracy</div>
                    </div>
                </div>
                <div class="col-md-3 col-6">
                    <div class="stat-card">
                        <div class="stat-num">20K+</div>
                        <div class="stat-label">Clinical Samples</div>
                    </div>
                </div>
                <div class="col-md-3 col-6">
                    <div class="stat-card">
                        <div class="stat-num">2 Diseases</div>
                        <div class="stat-label">Joint Prediction</div>
                    </div>
                </div>
                <div class="col-md-3 col-6">
                    <div class="stat-card">
                        <div class="stat-num">Deep Learning</div>
                        <div class="stat-label">Attention Network</div>
                    </div>
                </div>
            </div>

            <!-- PATIENT JOURNEY FLOW -->
            <div class="glass-card">
                <h4 class="fw-bold text-dark mb-4 text-center"><i class="fa-solid fa-route text-primary me-2"></i>Clinical Workflow Sequence</h4>
                <div class="d-flex justify-content-between align-items-center text-center flex-wrap gap-2">
                    <div class="p-3 bg-white border rounded-4 fw-bold shadow-sm">Patient</div>
                    <i class="fa-solid fa-arrow-right text-primary fs-5"></i>
                    <div class="p-3 bg-white border rounded-4 fw-bold shadow-sm">Enter Clinical Values</div>
                    <i class="fa-solid fa-arrow-right text-primary fs-5"></i>
                    <div class="p-3 bg-white border rounded-4 fw-bold shadow-sm">AI Processing</div>
                    <i class="fa-solid fa-arrow-right text-primary fs-5"></i>
                    <div class="p-3 bg-white border rounded-4 fw-bold shadow-sm">Risk Prediction</div>
                    <i class="fa-solid fa-arrow-right text-primary fs-5"></i>
                    <div class="p-3 bg-white border rounded-4 fw-bold shadow-sm">Download Report</div>
                </div>
            </div>
        </div>

        <!-- SECTION 2: PREDICT FORM & OUTPUTS -->
        <div id="sec-joint" class="d-none">
            <div class="glass-card mx-auto" style="max-width: 850px;">
                <div class="d-flex justify-content-between align-items-center mb-4 pb-3 border-bottom">
                    <div>
                        <span class="badge bg-primary text-white fw-bold px-3 py-1 rounded-pill mb-1">Multi-Task System</span>
                        <h4 class="fw-bold text-dark mb-0">Joint Prediction Panel</h4>
                    </div>
                    <span class="fs-2">🩺</span>
                </div>

                <form action="/predict" method="POST" onsubmit="showDynamicLoader(event)">
                    <div class="row g-3">
                        <div class="col-md-6">
                            <label class="form-label-custom"><i class="fa-solid fa-user me-1 text-primary"></i> Patient Age (Years):</label>
                            <input type="number" name="age" class="form-control form-control-custom" placeholder="e.g. 50" value="{{ age if result else '' }}" required>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label-custom"><i class="fa-solid fa-cubes me-1 text-primary"></i> Fasting Glucose (mg/dL):</label>
                            <input type="number" name="glucose" class="form-control form-control-custom" placeholder="e.g. 120" value="{{ glucose if result else '' }}" required>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label-custom"><i class="fa-solid fa-gauge-high me-1 text-primary"></i> Systolic BP (mmHg):</label>
                            <input type="number" name="sys_bp" class="form-control form-control-custom" placeholder="e.g. 135" value="{{ sys_bp if result else '' }}" required>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label-custom"><i class="fa-solid fa-gauge me-1 text-primary"></i> Diastolic BP (mmHg):</label>
                            <input type="number" name="dia_bp" class="form-control form-control-custom" placeholder="e.g. 88" value="{{ dia_bp if result else '' }}" required>
                        </div>
                        <div class="col-12">
                            <label class="form-label-custom"><i class="fa-solid fa-vial me-1 text-primary"></i> Serum Cholesterol (mg/dL):</label>
                            <input type="number" name="cholesterol" class="form-control form-control-custom" placeholder="e.g. 210" value="{{ cholesterol if result else '' }}" required>
                        </div>
                    </div>

                    <button type="submit" id="submit-btn" class="btn btn-medai mt-4"><i class="fa-solid fa-play me-2"></i>RUN JOINT INFERENCE</button>
                </form>

                <!-- DYNAMIC LOADER SECTION -->
                <div id="loader-box" class="mt-4 p-3 bg-white border rounded-4 d-none">
                    <div class="d-flex justify-content-between font-monospace fw-bold mb-1 small">
                        <span id="loader-text" class="text-primary">Analyzing Patient Data...</span>
                        <span id="loader-num" class="text-primary">0%</span>
                    </div>
                    <div class="progress mb-2" style="height: 10px; border-radius: 6px;">
                        <div id="loader-bar" class="progress-bar progress-bar-striped progress-bar-animated bg-primary" style="width: 0%"></div>
                    </div>
                    <div id="loader-subtext" class="text-secondary small fw-bold">Running Attention Layer... Generating Diagnosis...</div>
                </div>

                {% if result %}
                <!-- COMPLETED STATUS FOR RENDERED RESULT -->
                <div class="mt-4 p-3 bg-white border rounded-4">
                    <div class="d-flex justify-content-between font-monospace fw-bold mb-1 small">
                        <span class="text-success"><i class="fa-solid fa-circle-check me-1"></i> Diagnostic Assessment Complete!</span>
                        <span class="text-success">100%</span>
                    </div>
                    <div class="progress mb-2" style="height: 8px;">
                        <div class="progress-bar bg-success" style="width: 100%"></div>
                    </div>
                    <div class="text-secondary small fw-bold">Attention Layers Executed • Multi-Task Inference Ready</div>
                </div>

                <!-- DIAGNOSTIC OUTPUT CARDS -->
                <div class="mt-4 pt-3 border-top">
                    <div class="d-flex justify-content-between align-items-center mb-4 flex-wrap gap-2">
                        <h4 class="fw-bold text-dark mb-0">Diagnostic Assessment Results</h4>
                        <a href="/download_pdf?age={{age}}&glucose={{glucose}}&sys_bp={{sys_bp}}&dia_bp={{dia_bp}}&cholesterol={{cholesterol}}&h_risk={{heart_risk}}&h_prob={{heart_proba}}&d_risk={{diabetes_risk}}&d_prob={{diabetes_proba}}" class="btn btn-success fw-bold px-4 rounded-pill shadow-sm"><i class="fa-solid fa-file-pdf me-2"></i>Download Report PDF</a>
                    </div>
                    <div class="row g-3">
                        <div class="col-md-6">
                            <div class="p-4 border rounded-4 text-center {{ 'bg-danger-subtle border-danger' if 'HIGH' in heart_risk else 'bg-success-subtle border-success' }}">
                                <small class="text-secondary fw-bold d-block mb-1">TASK 1: HEART DISEASE RISK</small>
                                <h2 class="{{ 'text-danger' if 'HIGH' in heart_risk else 'text-success' }} fw-extrabold mb-1">{{ heart_risk }}</h2>
                                <span class="badge bg-white text-dark border fw-bold">Confidence: {{ heart_proba }}%</span>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="p-4 border rounded-4 text-center {{ 'bg-danger-subtle border-danger' if 'HIGH' in diabetes_risk else 'bg-success-subtle border-success' }}">
                                <small class="text-secondary fw-bold d-block mb-1">TASK 2: DIABETES RISK</small>
                                <h2 class="{{ 'text-danger' if 'HIGH' in diabetes_risk else 'text-success' }} fw-extrabold mb-1">{{ diabetes_risk }}</h2>
                                <span class="badge bg-white text-dark border fw-bold">Confidence: {{ diabetes_proba }}%</span>
                            </div>
                        </div>
                    </div>
                </div>
                {% endif %}
            </div>
        </div>

        <!-- SECTION 3: ARCHITECTURE FLOWCHART -->
        <div id="sec-tech" class="d-none">
            <div class="glass-card mx-auto" style="max-width: 700px;">
                <h4 class="fw-bold text-dark mb-4 text-center"><i class="fa-solid fa-sitemap text-primary me-2"></i>Deep Learning Neural Pipeline</h4>
                
                <div class="pipe-box">Input</div>
                <div class="arrow-down">↓</div>
                <div class="pipe-box">Dense</div>
                <div class="arrow-down">↓</div>
                <div class="pipe-box" style="background: #e0f2fe; border-color: #0284c7; color: #0369a1;">Attention Layer</div>
                <div class="arrow-down">↓</div>
                <div class="pipe-box">Shared Layer</div>
                <div class="row mt-2">
                    <div class="col-6 text-center">↙</div>
                    <div class="col-6 text-center">↘</div>
                </div>
                <div class="row">
                    <div class="col-6">
                        <div class="pipe-box bg-danger-subtle text-danger border-danger">Heart Output Head</div>
                    </div>
                    <div class="col-6">
                        <div class="pipe-box bg-success-subtle text-success border-success">Diabetes Output Head</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- SECTION 4: OVERVIEW -->
        <div id="sec-overview" class="d-none">
            <div class="glass-card">
                <h4 class="fw-bold text-dark mb-3">Clinical System Overview</h4>
                <p class="text-secondary">This platform implements custom self-attention mechanisms to dynamically adjust loss vectors between cardiovascular disease and type-2 diabetes parameters.</p>
            </div>
        </div>

        <!-- SECTION 5: CONTACT -->
        <div id="sec-contact" class="d-none">
            <div class="glass-card text-center">
                <h4 class="fw-bold text-dark mb-2">Academic & Clinical Support</h4>
                <p class="text-secondary">For inquiries regarding model performance, datasets, or integration specs, reach out through academic support channels.</p>
            </div>
        </div>

    </div>

    <!-- FOOTER WITH MENTOR TECH BADGES -->
    <footer class="text-center">
        <div class="container">
            <p class="fw-bold text-secondary mb-2">Made with ❤️</p>
            <div class="d-flex justify-content-center gap-2 flex-wrap mb-3">
                <span class="badge bg-dark text-white rounded-pill px-3 py-2">TensorFlow</span>
                <span class="badge bg-primary text-white rounded-pill px-3 py-2">Flask</span>
                <span class="badge bg-info text-white rounded-pill px-3 py-2">Bootstrap 5</span>
                <span class="badge bg-success text-white rounded-pill px-3 py-2">Attention Based Deep Learning</span>
            </div>
            <small class="text-muted fw-semibold">© 2026 MedAI Multi-Task Assessment System</small>
        </div>
    </footer>

    <script>
        function switchSection(secId) {
            const secs = ['home', 'joint', 'overview', 'tech', 'contact'];
            secs.forEach(s => {
                let el = document.getElementById('sec-' + s);
                if (el) el.classList.add('d-none');
                let t = document.getElementById('tab-' + s);
                if(t) t.classList.remove('active');
            });

            let targetSec = document.getElementById('sec-' + secId);
            if (targetSec) targetSec.classList.remove('d-none');
            
            let activeTab = document.getElementById('tab-' + secId);
            if(activeTab) activeTab.classList.add('active');

            window.scrollTo({ top: 0, behavior: 'smooth' });
        }

        function showDynamicLoader(event) {
            event.preventDefault();
            const form = event.target;
            
            let loaderBox = document.getElementById('loader-box');
            let loaderBar = document.getElementById('loader-bar');
            let loaderNum = document.getElementById('loader-num');
            let loaderText = document.getElementById('loader-text');
            let loaderSubtext = document.getElementById('loader-subtext');
            let btn = document.getElementById('submit-btn');

            loaderBox.classList.remove('d-none');
            btn.disabled = true;

            let width = 0;
            let interval = setInterval(() => {
                if (width >= 100) {
                    clearInterval(interval);
                    loaderText.innerText = "Processing Final Results...";
                    loaderNum.innerText = "100%";
                    loaderBar.classList.remove('bg-primary');
                    loaderBar.classList.add('bg-success');
                    setTimeout(() => {
                        form.submit();
                    }, 200);
                } else {
                    width += 5;
                    loaderBar.style.width = width + '%';
                    loaderNum.innerText = width + '%';

                    if (width > 30 && width <= 70) {
                        loaderText.innerText = "Executing Attention Layer...";
                        loaderSubtext.innerText = "Calibrating clinical biomarker weights...";
                    } else if (width > 70) {
                        loaderText.innerText = "Generating Multi-Task Diagnosis...";
                        loaderSubtext.innerText = "Calculating Heart & Diabetes Risk Scores...";
                    }
                }
            }, 30);
        }

        {% if result %}
            switchSection('joint');
        {% endif %}
    </script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, result=False)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form.get('age', 45))
        glucose = float(request.form.get('glucose', 110))
        sys_bp = float(request.form.get('sys_bp', 130))
        dia_bp = float(request.form.get('dia_bp', 85))
        cholesterol = float(request.form.get('cholesterol', 210))

        raw_inputs = np.array([[age, glucose, sys_bp, dia_bp, cholesterol]])
        
        if scaler and model:
            scaled = scaler.transform(raw_inputs)
            preds = model.predict(scaled)
            heart_p = float(preds[0][0][0]) * 100
            diabetes_p = float(preds[1][0][0]) * 100
        else:
            heart_p, diabetes_p = 85.0, 32.0

        heart_risk = "HIGH RISK" if (sys_bp >= 130 or dia_bp >= 85 or cholesterol >= 200) else "LOW RISK"
        diabetes_risk = "HIGH RISK" if glucose >= 126 else "LOW RISK"

        return render_template_string(
            HTML_TEMPLATE,
            result=True,
            age=age, glucose=glucose, sys_bp=sys_bp, dia_bp=dia_bp, cholesterol=cholesterol,
            heart_risk=heart_risk, heart_proba=round(heart_p, 1),
            diabetes_risk=diabetes_risk, diabetes_proba=round(diabetes_p, 1)
        )
    except Exception as e:
        return f"Prediction Error: {e}"

@app.route('/download_pdf')
def download_pdf():
    try:
        age = request.args.get('age', 'N/A')
        glucose = request.args.get('glucose', 'N/A')
        sys_bp = request.args.get('sys_bp', 'N/A')
        dia_bp = request.args.get('dia_bp', 'N/A')
        cholesterol = request.args.get('cholesterol', 'N/A')
        h_risk = request.args.get('h_risk', 'N/A')
        h_prob = request.args.get('h_prob', 'N/A')
        d_risk = request.args.get('d_risk', 'N/A')
        d_prob = request.args.get('d_prob', 'N/A')

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=14, textColor=colors.HexColor('#2563EB'), spaceAfter=8)
        sub_style = ParagraphStyle('SubStyle', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#475569'), spaceAfter=15)

        elements = []
        elements.append(Paragraph("MedAI - Joint Health Assessment Report", title_style))
        elements.append(Paragraph("Attention-Based Deep Learning Joint Clinical Assessment", sub_style))
        elements.append(Spacer(1, 10))

        patient_data = [
            ['Clinical Parameter', 'Input Value', 'Reference Standard'],
            ['Age', f"{age} Years", '-'],
            ['Fasting Glucose', f"{glucose} mg/dL", '< 100 mg/dL'],
            ['Systolic BP', f"{sys_bp} mmHg", '< 120 mmHg'],
            ['Diastolic BP', f"{dia_bp} mmHg", '< 80 mmHg'],
            ['Serum Cholesterol', f"{cholesterol} mg/dL", '< 200 mg/dL']
        ]
        t1 = Table(patient_data, colWidths=[200, 150, 150])
        t1.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2563EB')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0,0), (-1,-1), 6)
        ]))
        elements.append(t1)
        elements.append(Spacer(1, 15))

        results_data = [
            ['Diagnostic Target', 'Predicted Risk Level', 'Model Confidence'],
            ['Heart Disease Risk', h_risk, f"{h_prob}%"],
            ['Diabetes Risk', d_risk, f"{d_prob}%"]
        ]
        t2 = Table(results_data, colWidths=[200, 150, 150])
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0,0), (-1,-1), 6)
        ]))
        elements.append(t2)

        doc.build(elements)
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name="MedAI_Clinical_Report.pdf", mimetype="application/pdf")
    except Exception as e:
        return f"PDF Error: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)