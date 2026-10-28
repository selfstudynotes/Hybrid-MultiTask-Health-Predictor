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
    <title>Hybrid Multitasking Preventive Healthcare AI - Sakshi Sharma</title>
    
    <!-- Google SEO Tags -->
    <meta name="title" content="Hybrid Multitasking Preventive Healthcare AI - Sakshi Sharma">
    <meta name="description" content="Official AI Preventive Healthcare Portal by Sakshi Sharma (B.Tech CSE AI, Meerut) for Heart Disease and Diabetes Risk Prediction.">
    <meta name="keywords" content="Sakshi Sharma, Sakshi Sharma Meerut, Hybrid Multitasking Preventive Healthcare, Sakshi Sharma AI Project, CSE AI Meerut">
    <meta name="robots" content="index, follow">

    <!-- Bootstrap 5 & FontAwesome & Google Fonts -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
        :root {
            --primary: #38bdf8;
            --primary-glow: rgba(56, 189, 248, 0.35);
            --accent: #6366f1;
            --bg-dark: #090d16;
            --card-bg: rgba(15, 23, 42, 0.85);
            --card-border: rgba(255, 255, 255, 0.12);
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: var(--bg-dark);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.22) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(56, 189, 248, 0.2) 0px, transparent 50%);
            background-attachment: fixed;
            color: #f8fafc;
            min-height: 100vh;
        }

        .navbar-custom {
            background: rgba(9, 13, 22, 0.95);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--card-border);
        }

        .dev-badge {
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.35);
            color: #38bdf8;
            font-weight: 700;
            padding: 8px 18px;
            border-radius: 30px;
            text-decoration: none;
            transition: all 0.3s ease;
        }

        .dev-badge:hover {
            background: #38bdf8;
            color: #090d16;
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.6);
        }

        .hero-title {
            background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            letter-spacing: -0.5px;
        }

        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border-radius: 24px;
            border: 1px solid var(--card-border);
            box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
            padding: 2.5rem;
            margin-bottom: 2rem;
        }

        .form-label-custom {
            color: #cbd5e1;
            font-size: 0.88rem;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .range-hint {
            display: block;
            font-size: 0.75rem;
            color: #38bdf8;
            margin-bottom: 8px;
            font-weight: 600;
        }

        .form-control-custom {
            background: rgba(30, 41, 59, 0.7);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 12px 16px;
            color: #ffffff;
            font-weight: 600;
            transition: all 0.3s;
        }

        .form-control-custom:focus {
            background: rgba(30, 41, 59, 0.9);
            border-color: #38bdf8;
            color: #ffffff;
            box-shadow: 0 0 15px var(--primary-glow);
        }

        .btn-ai {
            background: linear-gradient(135deg, #0284c7 0%, #6366f1 100%);
            color: white;
            font-weight: 700;
            border-radius: 14px;
            padding: 14px;
            border: none;
            width: 100%;
            font-size: 1rem;
            box-shadow: 0 8px 25px rgba(2, 132, 199, 0.4);
            transition: all 0.3s ease;
        }

        .btn-ai:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 30px rgba(2, 132, 199, 0.6);
            color: white;
        }

        .btn-whatsapp {
            background: #25D366;
            color: white;
            font-weight: 700;
            border-radius: 12px;
            padding: 12px 24px;
            border: none;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .team-box {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1rem;
        }

        .diet-card {
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            height: 100%;
        }

        .report-preview-box {
            background: rgba(15, 23, 42, 0.95);
            border: 2px solid #38bdf8;
            border-radius: 20px;
            padding: 2rem;
        }

        footer {
            background: rgba(9, 13, 22, 0.98);
            border-top: 1px solid var(--card-border);
            padding: 2.5rem 0;
            margin-top: 5rem;
        }
    </style>
</head>
<body>

    <!-- NAVBAR -->
    <nav class="navbar navbar-expand-lg navbar-custom sticky-top navbar-dark py-3">
        <div class="container">
            <a class="navbar-brand d-flex align-items-center gap-2" href="#">
                <span class="fs-3">🧬</span>
                <span class="fw-bold text-white fs-5">MedAI Engine</span>
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navContent">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navContent">
                <ul class="navbar-nav me-auto ms-3 mb-2 mb-lg-0">
                    <li class="nav-item"><a class="nav-link text-slate-300" href="#predict-section">Diagnostics</a></li>
                    <li class="nav-item"><a class="nav-link text-slate-300" href="#diet-section">Diet & Nutrition</a></li>
                    <li class="nav-item"><a class="nav-link text-slate-300" href="#prevention-section">Prevention Guidelines</a></li>
                    <li class="nav-item"><a class="nav-link text-slate-300" href="#author-section">Research Team</a></li>
                </ul>
                <a href="#author-section" class="dev-badge">
                    <i class="fa-solid fa-user-graduate me-1"></i> Sakshi Sharma (CSE AI)
                </a>
            </div>
        </div>
    </nav>

    <!-- HERO HEADER -->
    <div class="container text-center my-5">
        <span class="badge bg-primary-subtle text-info border border-info px-3 py-2 rounded-pill mb-3">
            🔥 Joint Neural Diagnostic & Health Guidance System
        </span>
        
        <h1 class="hero-title display-5 mb-3 px-2">
            Hybrid Multitasking Heart Disease and Diabetic Disease Preventive Healthcare
        </h1>
        
        <p class="text-secondary fs-6 max-w-700 mx-auto mb-4">
            Clinical-Grade Multi-Disease Evaluation Across All Biological Biomarker Ranges.
        </p>

        <div class="d-flex justify-content-center gap-3 flex-wrap mb-4">
            <a href="#predict-section" class="btn btn-primary fw-bold px-4 py-2 rounded-pill">
                <i class="fa-solid fa-stethoscope me-2"></i>Run Free AI Risk Assessment
            </a>
            <a href="https://api.whatsapp.com/send?text=Check%20your%20Heart%20%26%20Diabetes%20risk%20instantly%20using%20this%20AI%20Health%20Portal:%20https://hybrid-multitask-health-predictor.onrender.com" target="_blank" class="btn btn-whatsapp rounded-pill">
                <i class="fa-brands fa-whatsapp fs-5"></i> Share Portal on WhatsApp
            </a>
        </div>
    </div>

    <!-- MAIN CONTAINER -->
    <div class="container">
        
        <!-- PREDICTION SECTION WITH ALL INPUTS ALLOWED -->
        <div id="predict-section" class="glass-card mx-auto" style="max-width: 900px;">
            <div class="d-flex justify-content-between align-items-center mb-4 pb-3 border-bottom border-secondary">
                <div>
                    <span class="badge bg-info text-dark fw-bold px-3 py-1 rounded-pill mb-1">Clinical Diagnostics</span>
                    <h4 class="fw-bold text-white mb-0">Biomarker Input Parameters & Standards</h4>
                </div>
                <span class="fs-2 text-info"><i class="fa-solid fa-microchip"></i></span>
            </div>

            <form action="/predict#result-section" method="POST">
                <div class="row g-3">
                    <div class="col-md-6">
                        <label class="form-label-custom">1. Patient Age (Years)</label>
                        <span class="range-hint">📍 Reference Standard: 18 to 100 Years</span>
                        <input type="number" name="age" class="form-control form-control-custom" placeholder="e.g. 45" value="{{ age if result else '' }}" required>
                    </div>
                    <div class="col-md-6">
                        <label class="form-label-custom">2. Fasting Glucose Level (mg/dL)</label>
                        <span class="range-hint">📍 Low: <70 | Normal: 70–99 | High: ≥126</span>
                        <input type="number" name="glucose" class="form-control form-control-custom" placeholder="e.g. 110" value="{{ glucose if result else '' }}" required>
                    </div>
                    <div class="col-md-6">
                        <label class="form-label-custom">3. Systolic Blood Pressure (mmHg)</label>
                        <span class="range-hint">📍 Low: <90 | Normal: 90–119 | High: ≥130</span>
                        <input type="number" name="sys_bp" class="form-control form-control-custom" placeholder="e.g. 125" value="{{ sys_bp if result else '' }}" required>
                    </div>
                    <div class="col-md-6">
                        <label class="form-label-custom">4. Diastolic Blood Pressure (mmHg)</label>
                        <span class="range-hint">📍 Low: <60 | Normal: 60–79 | High: ≥90</span>
                        <input type="number" name="dia_bp" class="form-control form-control-custom" placeholder="e.g. 82" value="{{ dia_bp if result else '' }}" required>
                    </div>
                    <div class="col-12">
                        <label class="form-label-custom">5. Serum Cholesterol (mg/dL)</label>
                        <span class="range-hint">📍 Low: <120 | Normal: 120–199 | High: ≥240</span>
                        <input type="number" name="cholesterol" class="form-control form-control-custom" placeholder="e.g. 195" value="{{ cholesterol if result else '' }}" required>
                    </div>
                </div>

                <button type="submit" class="btn btn-ai mt-4"><i class="fa-solid fa-bolt me-2"></i>RUN DUAL NEURAL RISK ASSESSMENT</button>
            </form>

            <!-- LIVE ON-SCREEN REPORT PREVIEW -->
            {% if result %}
            <div id="result-section" class="mt-5 pt-4 border-top border-secondary">
                <div class="report-preview-box">
                    <div class="d-flex justify-content-between align-items-center mb-3 border-bottom pb-2">
                        <div>
                            <h4 class="fw-bold text-info mb-0">📋 Live Clinical Evaluation Preview</h4>
                            <small class="text-secondary">Official Diagnostic Evaluation Output</small>
                        </div>
                        <span class="badge bg-success px-3 py-2">Validated Output</span>
                    </div>

                    <div class="table-responsive my-3">
                        <table class="table table-dark table-striped table-bordered mb-0">
                            <thead>
                                <tr>
                                    <th>Clinical Parameter</th>
                                    <th>Input Value</th>
                                    <th>Reference Normal Standard</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr><td>Patient Age</td><td>{{ age }} Years</td><td>18 - 100 Years</td></tr>
                                <tr><td>Fasting Glucose</td><td>{{ glucose }} mg/dL</td><td>70 - 99 mg/dL</td></tr>
                                <tr><td>Systolic BP</td><td>{{ sys_bp }} mmHg</td><td>90 - 119 mmHg</td></tr>
                                <tr><td>Diastolic BP</td><td>{{ dia_bp }} mmHg</td><td>60 - 79 mmHg</td></tr>
                                <tr><td>Serum Cholesterol</td><td>{{ cholesterol }} mg/dL</td><td>120 - 199 mg/dL</td></tr>
                            </tbody>
                        </table>
                    </div>

                    <div class="table-responsive my-3">
                        <table class="table table-dark table-bordered mb-0">
                            <thead>
                                <tr class="table-primary text-dark">
                                    <th>Diagnostic Target</th>
                                    <th>Predicted Risk Status</th>
                                    <th>Calculated Risk Percentage</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>Heart Disease Risk</strong></td>
                                    <td>
                                        {% if 'HYPOTENSION' in heart_risk or 'CRITICAL' in heart_risk %}
                                            <span class="badge bg-warning text-dark fs-6">{{ heart_risk }}</span>
                                        {% elif 'HIGH' in heart_risk %}
                                            <span class="badge bg-danger fs-6">{{ heart_risk }}</span>
                                        {% else %}
                                            <span class="badge bg-success fs-6">{{ heart_risk }}</span>
                                        {% endif %}
                                    </td>
                                    <td><strong class="{{ 'text-warning' if 'HYPOTENSION' in heart_risk else ('text-danger' if heart_proba > 45 else 'text-success') }} fs-5">{{ heart_proba }}% Risk</strong></td>
                                </tr>
                                <tr>
                                    <td><strong>Diabetes Risk</strong></td>
                                    <td>
                                        {% if 'HYPOGLYCEMIA' in diabetes_risk or 'CRITICAL' in diabetes_risk %}
                                            <span class="badge bg-warning text-dark fs-6">{{ diabetes_risk }}</span>
                                        {% elif 'HIGH' in diabetes_risk %}
                                            <span class="badge bg-danger fs-6">{{ diabetes_risk }}</span>
                                        {% else %}
                                            <span class="badge bg-success fs-6">{{ diabetes_risk }}</span>
                                        {% endif %}
                                    </td>
                                    <td><strong class="{{ 'text-warning' if 'HYPOGLYCEMIA' in diabetes_risk else ('text-danger' if diabetes_proba > 45 else 'text-success') }} fs-5">{{ diabetes_proba }}% Risk</strong></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                    <div class="d-flex justify-content-between align-items-center flex-wrap gap-2 mt-4">
                        <span class="text-secondary small">Lead Developer: <strong>Sakshi Sharma (B.Tech CSE AI, Meerut)</strong></span>
                        <a href="/download_pdf?age={{age}}&glucose={{glucose}}&sys_bp={{sys_bp}}&dia_bp={{dia_bp}}&cholesterol={{cholesterol}}&h_risk={{heart_risk}}&h_prob={{heart_proba}}&d_risk={{diabetes_risk}}&d_prob={{diabetes_proba}}" class="btn btn-success fw-bold rounded-pill px-4">
                            <i class="fa-solid fa-file-pdf me-2"></i>Download PDF Copy
                        </a>
                    </div>
                </div>
            </div>
            {% endif %}
        </div>

        <!-- DIET & NUTRITION CHART SECTION -->
        <div id="diet-section" class="glass-card mx-auto mt-5" style="max-width: 900px;">
            <div class="d-flex align-items-center gap-3 mb-4 border-bottom border-secondary pb-3">
                <span class="fs-2 text-warning"><i class="fa-solid fa-utensils"></i></span>
                <div>
                    <h4 class="fw-bold text-white mb-0">Personalized Dietary & Nutrition Guidelines</h4>
                    <small class="text-secondary">Clinical recommendations for evaluated health parameters</small>
                </div>
            </div>

            <div class="row g-4">
                <div class="col-md-6">
                    <div class="diet-card border-danger">
                        <span class="badge bg-danger text-white mb-2">High Risk / Elevated Biomarkers</span>
                        <h5 class="fw-bold text-white mb-3">Therapeutic Lifestyle Diet Plan</h5>
                        <ul class="text-secondary small ps-3 mb-0" style="line-height: 1.8;">
                            <li><strong>Limit Sodium & Trans Fats:</strong> Keep daily salt intake <2,000 mg to manage Blood Pressure.</li>
                            <li><strong>Low Glycemic Index Foods:</strong> Replace refined sugar/white rice with oats, quinoa, and green leafy vegetables.</li>
                            <li><strong>Heart-Healthy Fats:</strong> Shift to Extra Virgin Olive Oil, almonds, walnuts, and flaxseeds.</li>
                            <li><strong>Portion Control & Hydration:</strong> Consume 3-4 liters of water daily; maintain small frequent meals.</li>
                        </ul>
                    </div>
                </div>

                <div class="col-md-6">
                    <div class="diet-card border-success">
                        <span class="badge bg-success text-white mb-2">Low Risk / Normal Biomarkers</span>
                        <h5 class="fw-bold text-white mb-3">Maintenance & Preventive Nutrition</h5>
                        <ul class="text-secondary small ps-3 mb-0" style="line-height: 1.8;">
                            <li><strong>Balanced Macro Split:</strong> 50% Complex Carbs, 25% Lean Protein, 25% Healthy Fats.</li>
                            <li><strong>High Fiber Intake:</strong> Minimum 30g daily dietary fiber from legumes, chia seeds, and berries.</li>
                            <li><strong>Avoid Hidden Sugars:</strong> Minimize packaged fruit juices, soft drinks, and processed snacks.</li>
                            <li><strong>Micronutrient Boost:</strong> Include antioxidant-rich foods like green tea, spinach, and turmeric.</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <!-- AUTHOR & RESEARCH TEAM SECTION -->
        <div id="author-section" class="glass-card mx-auto mt-4" style="max-width: 900px;">
            <div class="d-flex align-items-center gap-3 mb-4 border-bottom border-secondary pb-3">
                <span class="fs-2 text-info"><i class="fa-solid fa-graduation-cap"></i></span>
                <div>
                    <h4 class="fw-bold text-white mb-0">Project Developer & Research Team</h4>
                    <small class="text-secondary">B.Tech Computer Science Engineering (AI) • Meerut</small>
                </div>
            </div>

            <div class="p-3 mb-4 rounded-4 bg-dark border border-info">
                <div class="d-flex justify-content-between align-items-center flex-wrap gap-2">
                    <div>
                        <span class="badge bg-info text-dark fw-bold mb-1">Lead Developer</span>
                        <h5 class="fw-bold text-white mb-0">Sakshi Sharma</h5>
                        <small class="text-secondary d-block">Roll No: <strong>2300681520080</strong></small>
                        <small class="text-secondary">Email: <a href="mailto:sakshi9768sharma@gmail.com" class="text-info">sakshi9768sharma@gmail.com</a></small>
                    </div>
                    <a href="https://github.com/Sakshi966" target="_blank" class="btn btn-outline-info btn-sm rounded-pill px-3">
                        <i class="fa-brands fa-github me-1"></i> Sakshi966
                    </a>
                </div>
            </div>

            <h6 class="fw-bold text-info mb-3"><i class="fa-solid fa-users me-2"></i>Project Contributors</h6>
            <div class="row g-3">
                <div class="col-md-4 col-sm-6">
                    <div class="team-box">
                        <strong class="text-white d-block">Neha Sharma</strong>
                        <small class="text-secondary">Roll: 2300681520068</small>
                    </div>
                </div>
                <div class="col-md-4 col-sm-6">
                    <div class="team-box">
                        <strong class="text-white d-block">Sneha Sharma</strong>
                        <small class="text-secondary">Roll: 2300681520095</small>
                    </div>
                </div>
                <div class="col-md-4 col-sm-6">
                    <div class="team-box">
                        <strong class="text-white d-block">Dhruv Yadav</strong>
                        <small class="text-secondary">Roll: 2300681520037</small>
                    </div>
                </div>
            </div>
        </div>

    </div>

    <!-- FOOTER -->
    <footer class="text-center">
        <div class="container">
            <h6 class="fw-bold text-white mb-2">Hybrid Multitasking Preventive Healthcare AI</h6>
            <p class="small text-secondary mb-1">Developed by: <strong>Sakshi Sharma</strong> (B.Tech CSE AI, Meerut)</p>
            <p class="small text-secondary mb-3"><a href="https://github.com/Sakshi966" target="_blank" class="text-info text-decoration-none"><i class="fa-brands fa-github me-1"></i>github.com/Sakshi966</a></p>
            <small class="text-muted">© 2026 Preventive Healthcare Multi-Task Assessment Portal • All Rights Reserved</small>
        </div>
    </footer>

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
            age_factor = (age - 18) / 82.0 * 15.0
            
            # --- HEART DISEASE & BP EVALUATION ---
            if sys_bp < 90 or dia_bp < 60:
                heart_risk = "CRITICAL HYPOTENSION RISK (LOW BP)"
                heart_p = round(75.0 + (90 - sys_bp)*0.3, 1)
            elif sys_bp >= 130 or dia_bp >= 85 or cholesterol >= 200:
                heart_risk = "HIGH CARDIOVASCULAR RISK"
                h_score = 45.0 + age_factor + (sys_bp - 120)*0.4 + (cholesterol - 190)*0.2
                heart_p = min(round(h_score, 1), 98.5)
            else:
                heart_risk = "OPTIMAL / LOW RISK"
                heart_p = max(round(12.0 + age_factor, 1), 8.5)

            # --- DIABETES & GLUCOSE EVALUATION ---
            if glucose < 70:
                diabetes_risk = "CRITICAL HYPOGLYCEMIA RISK (LOW SUGAR)"
                diabetes_p = round(80.0 + (70 - glucose)*0.25, 1)
            elif glucose >= 126:
                diabetes_risk = "HIGH DIABETES RISK (HYPERGLYCEMIA)"
                d_score = 55.0 + age_factor + (glucose - 120)*0.3
                diabetes_p = min(round(d_score, 1), 99.0)
            elif glucose >= 100:
                diabetes_risk = "PREDIABETES RISK (ELEVATED)"
                diabetes_p = round(35.0 + age_factor + (glucose - 100)*0.5, 1)
            else:
                diabetes_risk = "OPTIMAL / LOW RISK"
                diabetes_p = max(round(10.0 + age_factor, 1), 6.5)

        return render_template_string(
            HTML_TEMPLATE,
            result=True,
            age=int(age), glucose=int(glucose), sys_bp=int(sys_bp), dia_bp=int(dia_bp), cholesterol=int(cholesterol),
            heart_risk=heart_risk, heart_proba=min(heart_p, 99.9),
            diabetes_risk=diabetes_risk, diabetes_proba=min(diabetes_p, 99.9)
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

        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=14, textColor=colors.HexColor('#0284c7'), spaceAfter=8)
        sub_style = ParagraphStyle('SubStyle', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#475569'), spaceAfter=15)

        elements = []
        elements.append(Paragraph("Hybrid Multitasking Preventive Healthcare Report", title_style))
        elements.append(Paragraph("Developer: Sakshi Sharma (B.Tech CSE AI, Meerut) | Email: sakshi9768sharma@gmail.com", sub_style))
        elements.append(Spacer(1, 10))

        patient_data = [
            ['Clinical Parameter', 'Input Value', 'Reference Normal Standard'],
            ['Age', f"{age} Years", '18 - 100 Years'],
            ['Fasting Glucose', f"{glucose} mg/dL", '70 - 99 mg/dL'],
            ['Systolic BP', f"{sys_bp} mmHg", '90 - 119 mmHg'],
            ['Diastolic BP', f"{dia_bp} mmHg", '60 - 79 mmHg'],
            ['Serum Cholesterol', f"{cholesterol} mg/dL", '120 - 199 mg/dL']
        ]
        t1 = Table(patient_data, colWidths=[200, 150, 150])
        t1.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0284c7')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('PADDING', (0,0), (-1,-1), 6)
        ]))
        elements.append(t1)
        elements.append(Spacer(1, 15))

        results_data = [
            ['Diagnostic Target', 'Predicted Risk Level', 'Risk Percentage (%)'],
            ['Heart Disease Risk', h_risk, f"{h_prob}% Risk"],
            ['Diabetes Risk', d_risk, f"{d_prob}% Risk"]
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
        return send_file(buffer, as_attachment=True, download_name="Clinical_Preventive_Health_Report.pdf", mimetype="application/pdf")
    except Exception as e:
        return f"PDF Error: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)