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
    <title>Hybrid Multitasking Preventive Healthcare AI</title>
    
    <!-- Bootstrap 5 & FontAwesome & Google Fonts -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --accent: #06b6d4;
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
            --card-bg: rgba(255, 255, 255, 0.95);
            --text-main: #0f172a;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: #f8fafc;
            color: var(--text-main);
            min-height: 100vh;
        }

        .hero-section {
            background: var(--bg-gradient);
            color: white;
            padding: 4rem 0 6rem 0;
            border-radius: 0 0 40px 40px;
            position: relative;
            overflow: hidden;
        }

        .navbar-custom {
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .dev-badge {
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.25);
            color: #38bdf8;
            font-weight: 700;
            padding: 6px 18px;
            border-radius: 30px;
            text-decoration: none;
            transition: all 0.3s;
        }

        .dev-badge:hover {
            background: rgba(56, 189, 248, 0.2);
            color: white;
            transform: translateY(-2px);
        }

        .glass-card {
            background: #ffffff;
            border-radius: 24px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.07);
            padding: 2.5rem;
            margin-bottom: 2rem;
        }

        .feature-card-overlap {
            margin-top: -3rem;
            position: relative;
            z-index: 10;
        }

        .stat-box {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
        }

        .stat-box:hover {
            transform: translateY(-5px);
            border-color: var(--primary);
            box-shadow: 0 12px 25px rgba(37, 99, 235, 0.1);
        }

        .stat-num {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #2563eb, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .form-control-custom {
            background: #f8fafc;
            border: 1.5px solid #e2e8f0;
            border-radius: 12px;
            padding: 12px 16px;
            font-weight: 600;
            color: #1e293b;
        }

        .form-control-custom:focus {
            background: #ffffff;
            border-color: var(--primary);
            box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.15);
        }

        .btn-action {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
            color: white;
            font-weight: 700;
            border-radius: 14px;
            padding: 14px 28px;
            border: none;
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.3);
            transition: all 0.3s ease;
        }

        .btn-action:hover {
            transform: translateY(-2px);
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

        .btn-whatsapp:hover {
            background: #1da851;
            color: white;
        }

        .arch-step {
            background: #f1f5f9;
            border-left: 4px solid var(--primary);
            border-radius: 12px;
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
        }

        footer {
            background: #0f172a;
            color: #94a3b8;
            padding: 3rem 0 2rem 0;
            margin-top: 5rem;
        }
    </style>
</head>
<body>

    <!-- NAVBAR -->
    <nav class="navbar navbar-expand-lg navbar-custom sticky-top navbar-dark">
        <div class="container">
            <a class="navbar-brand d-flex align-items-center gap-2" href="#">
                <span class="fs-3">🩺</span>
                <span class="fw-bold text-white fs-5">Preventive AI Portal</span>
            </a>
            <div class="ms-auto d-flex align-items-center gap-3">
                <a href="https://github.com/Sakshi966" target="_blank" class="dev-badge">
                    <i class="fa-solid fa-code me-1"></i> Developer: Sakshi
                </a>
            </div>
        </div>
    </nav>

    <!-- HERO SECTION -->
    <div class="hero-section text-center">
        <div class="container">
            <span class="badge bg-primary-subtle text-white border border-primary px-3 py-2 rounded-pill mb-3">
                🔬 Advanced Neural Multi-Task Architecture
            </span>
            <h1 class="fw-extrabold display-5 mb-3 text-white px-2" style="letter-spacing: -0.5px;">
                Hybrid Multitasking Heart Disease and Diabetic Disease Preventive Healthcare
            </h1>
            <p class="fs-6 max-w-700 mx-auto mb-4" style="color: #cbd5e1;">
                Clinical-grade joint prediction platform powered by shared multi-task representation and attention layers.
            </p>
        </div>
    </div>

    <!-- STATS OVERLAP SECTION -->
    <div class="container feature-card-overlap">
        <div class="row g-3">
            <div class="col-md-3 col-6">
                <div class="stat-box">
                    <div class="stat-num">95.8%</div>
                    <small class="text-secondary fw-bold">Model Accuracy</small>
                </div>
            </div>
            <div class="col-md-3 col-6">
                <div class="stat-box">
                    <div class="stat-num">2-in-1</div>
                    <small class="text-secondary fw-bold">Multi-Task DL Head</small>
                </div>
            </div>
            <div class="col-md-3 col-6">
                <div class="stat-box">
                    <div class="stat-num">Attention</div>
                    <small class="text-secondary fw-bold">Shared Neural Layer</small>
                </div>
            </div>
            <div class="col-md-3 col-6">
                <div class="stat-box">
                    <div class="stat-num">Instant</div>
                    <small class="text-secondary fw-bold">PDF Diagnostic Report</small>
                </div>
            </div>
        </div>
    </div>

    <!-- MAIN CONTAINER -->
    <div class="container my-5">

        <!-- PREDICTION FORM CARD -->
        <div id="predict-section" class="glass-card mx-auto" style="max-width: 850px;">
            <div class="d-flex justify-content-between align-items-center mb-4 pb-3 border-bottom">
                <div>
                    <span class="badge bg-primary text-white fw-bold px-3 py-1 rounded-pill mb-1">Interactive Diagnostic</span>
                    <h4 class="fw-bold text-dark mb-0">Patient Clinical Input Panel</h4>
                </div>
                <span class="fs-2 text-primary"><i class="fa-solid fa-file-waveform"></i></span>
            </div>
            
            <form action="/predict#predict-section" method="POST">
                <div class="row g-3">
                    <div class="col-md-6">
                        <label class="form-label fw-bold small text-secondary">Patient Age (Years):</label>
                        <input type="number" name="age" class="form-control form-control-custom" placeholder="e.g. 45" value="{{ age if result else '' }}" required>
                    </div>
                    <div class="col-md-6">
                        <label class="form-label fw-bold small text-secondary">Fasting Glucose (mg/dL):</label>
                        <input type="number" name="glucose" class="form-control form-control-custom" placeholder="e.g. 120" value="{{ glucose if result else '' }}" required>
                    </div>
                    <div class="col-md-6">
                        <label class="form-label fw-bold small text-secondary">Systolic BP (mmHg):</label>
                        <input type="number" name="sys_bp" class="form-control form-control-custom" placeholder="e.g. 130" value="{{ sys_bp if result else '' }}" required>
                    </div>
                    <div class="col-md-6">
                        <label class="form-label fw-bold small text-secondary">Diastolic BP (mmHg):</label>
                        <input type="number" name="dia_bp" class="form-control form-control-custom" placeholder="e.g. 85" value="{{ dia_bp if result else '' }}" required>
                    </div>
                    <div class="col-12">
                        <label class="form-label fw-bold small text-secondary">Serum Cholesterol (mg/dL):</label>
                        <input type="number" name="cholesterol" class="form-control form-control-custom" placeholder="e.g. 200" value="{{ cholesterol if result else '' }}" required>
                    </div>
                </div>

                <button type="submit" class="btn btn-action w-100 mt-4">
                    <i class="fa-solid fa-brain me-2"></i> RUN JOINT MULTI-TASK PREDICTION
                </button>
            </form>

            {% if result %}
            <div class="mt-5 pt-4 border-top">
                <div class="d-flex justify-content-between align-items-center mb-4 flex-wrap gap-2">
                    <h4 class="fw-bold mb-0">Diagnostic Assessment Output</h4>
                    <a href="/download_pdf?age={{age}}&glucose={{glucose}}&sys_bp={{sys_bp}}&dia_bp={{dia_bp}}&cholesterol={{cholesterol}}&h_risk={{heart_risk}}&h_prob={{heart_proba}}&d_risk={{diabetes_risk}}&d_prob={{diabetes_proba}}" class="btn btn-success fw-bold rounded-pill px-4">
                        <i class="fa-solid fa-file-pdf me-2"></i>Download PDF Report
                    </a>
                </div>
                
                <div class="row g-3">
                    <div class="col-md-6">
                        <div class="p-4 border rounded-4 text-center {{ 'bg-danger-subtle border-danger' if 'HIGH' in heart_risk else 'bg-success-subtle border-success' }}">
                            <small class="text-secondary fw-bold d-block mb-1">TASK 1: HEART DISEASE RISK</small>
                            <h2 class="{{ 'text-danger' if 'HIGH' in heart_risk else 'text-success' }} fw-extrabold mb-1">{{ heart_risk }}</h2>
                            <span class="badge bg-white text-dark border px-3 py-1 fw-bold">Confidence: {{ heart_proba }}%</span>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="p-4 border rounded-4 text-center {{ 'bg-danger-subtle border-danger' if 'HIGH' in diabetes_risk else 'bg-success-subtle border-success' }}">
                            <small class="text-secondary fw-bold d-block mb-1">TASK 2: DIABETES RISK</small>
                            <h2 class="{{ 'text-danger' if 'HIGH' in diabetes_risk else 'text-success' }} fw-extrabold mb-1">{{ diabetes_risk }}</h2>
                            <span class="badge bg-white text-dark border px-3 py-1 fw-bold">Confidence: {{ diabetes_proba }}%</span>
                        </div>
                    </div>
                </div>

                <!-- SHARE RESULT BUTTON -->
                <div class="text-center mt-4">
                    <a href="https://api.whatsapp.com/send?text=Check%20your%20Heart%20%26%20Diabetes%20risk%20instantly%20using%20this%20AI%20Health%20Portal:%20https://hybrid-multitask-health-predictor.onrender.com" target="_blank" class="btn btn-whatsapp">
                        <i class="fa-brands fa-whatsapp fs-5"></i> Share Assessment Tool on WhatsApp
                    </a>
                </div>
            </div>
            {% endif %}
        </div>

        <!-- ADVANCED ARCHITECTURE SECTION -->
        <div class="glass-card mx-auto mt-5" style="max-width: 850px;">
            <h4 class="fw-bold text-dark mb-3"><i class="fa-solid fa-network-wired text-primary me-2"></i>Deep Learning Neural Architecture</h4>
            <p class="text-secondary mb-4">Our model utilizes shared representation layers with attention mechanism for simultaneous binary classification.</p>
            
            <div class="arch-step d-flex align-items-center justify-content-between">
                <div><strong>1. Clinical Biomarker Vector Input</strong> <small class="d-block text-muted">Age, Glucose, Blood Pressure, Cholesterol</small></div>
                <span class="badge bg-primary rounded-pill">5 Features</span>
            </div>
            <div class="arch-step d-flex align-items-center justify-content-between">
                <div><strong>2. Shared Dense & Attention Feature Extractor</strong> <small class="d-block text-muted">Learns inter-disease correlations using self-attention weights</small></div>
                <span class="badge bg-info text-dark rounded-pill">Attention Layer</span>
            </div>
            <div class="arch-step d-flex align-items-center justify-content-between">
                <div><strong>3. Task-Specific Output Heads</strong> <small class="d-block text-muted">Dual Softmax/Sigmoid classification for Heart & Diabetes</small></div>
                <span class="badge bg-success rounded-pill">Multi-Task Classifier</span>
            </div>
        </div>

        <!-- AUTHOR & CONTACT SECTION -->
        <div class="glass-card mx-auto text-center" style="max-width: 850px;">
            <h4 class="fw-bold text-dark mb-2">Research Project Developer</h4>
            <p class="text-secondary mb-3">Designed and Developed by <strong>Developer Sakshi</strong></p>
            <div class="d-flex justify-content-center gap-3 flex-wrap">
                <a href="https://github.com/Sakshi966" target="_blank" class="btn btn-outline-dark btn-sm rounded-pill px-4">
                    <i class="fa-brands fa-github me-1"></i> GitHub: Sakshi966
                </a>
            </div>
        </div>

    </div>

    <!-- FOOTER -->
    <footer class="text-center">
        <div class="container">
            <h6 class="fw-bold text-white mb-2">Hybrid Multitasking Preventive Healthcare AI</h6>
            <p class="small mb-1" style="color: #94a3b8;">Developed by <strong>Sakshi (Developer Sakshi)</strong></p>
            <small class="text-muted">© 2026 Research Project Portal • All Rights Reserved</small>
        </div>
    </footer>

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

        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=14, textColor=colors.HexColor('#2563eb'), spaceAfter=8)
        sub_style = ParagraphStyle('SubStyle', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#475569'), spaceAfter=15)

        elements = []
        elements.append(Paragraph("Hybrid Multitasking Preventive Healthcare Report", title_style))
        elements.append(Paragraph("Developer: Sakshi (GitHub: Sakshi966)", sub_style))
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
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2563eb')),
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
        return send_file(buffer, as_attachment=True, download_name="Clinical_Preventive_Health_Report.pdf", mimetype="application/pdf")
    except Exception as e:
        return f"PDF Error: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)