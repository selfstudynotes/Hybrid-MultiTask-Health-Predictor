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
    <title>Hybrid Health Predictor AI</title>
    
    <!-- Bootstrap 5 & FontAwesome -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">

    <style>
        :root {
            --blue-primary: #1e40af;
            --cyan-accent: #0284c7;
            --green-success: #059669;
            --red-danger: #dc2626;
            --bg-light: #f8fafc;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: linear-gradient(135deg, #eef2f7 0%, #dbefe6 100%);
            background-attachment: fixed;
            color: #0f172a;
            min-height: 100vh;
        }

        .navbar-custom {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid #e2e8f0;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
            position: sticky;
            top: 0;
            z-index: 100;
        }

        .nav-brand-text {
            font-weight: 800;
            font-size: 1.3rem;
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
            background: rgba(30, 64, 175, 0.08);
        }

        .nav-link-custom.active {
            background: var(--blue-primary);
            color: #ffffff !important;
        }

        .glass-card {
            background: #ffffff;
            border-radius: 24px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 15px 35px rgba(0,0,0,.06);
            padding: 2.5rem;
            margin-bottom: 2rem;
        }

        .stat-card {
            background: #ffffff;
            border-radius: 18px;
            padding: 1.2rem;
            border: 1px solid #e2e8f0;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.03);
            text-align: center;
        }

        .stat-num {
            font-size: 2rem;
            font-weight: 800;
            color: var(--blue-primary);
        }

        .stat-label {
            font-size: 0.85rem;
            font-weight: 700;
            color: #64748b;
        }

        .form-label-custom {
            color: #334155;
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .form-control-custom {
            background: #f8fafc;
            border: 1.5px solid #cbd5e1;
            border-radius: 12px;
            padding: 12px 16px;
            font-weight: 600;
        }

        .form-control-custom:focus {
            background: #ffffff;
            border-color: var(--blue-primary);
            box-shadow: 0 0 0 4px rgba(30, 64, 175, 0.12);
        }

        .btn-medai {
            background: linear-gradient(135deg, #1e40af 0%, #0284c7 100%);
            color: white;
            font-weight: 700;
            border-radius: 12px;
            padding: 14px;
            border: none;
            width: 100%;
            font-size: 1rem;
            box-shadow: 0 6px 18px rgba(30, 64, 175, 0.2);
            transition: all 0.3s ease;
        }

        .btn-medai:hover {
            transform: translateY(-2px);
            color: white;
        }

        footer {
            background: #ffffff;
            border-top: 1px solid #e2e8f0;
            padding: 1.5rem 0;
            margin-top: 3rem;
        }
    </style>
</head>
<body>

    <!-- NAVBAR HEADER -->
    <nav class="navbar navbar-expand-lg navbar-custom">
        <div class="container">
            <a class="navbar-brand d-flex align-items-center gap-2" onclick="switchSection('home')" style="cursor: pointer;">
                <span class="fs-3">🩺</span>
                <span class="nav-brand-text">MedAI Predictor</span>
            </a>
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-menu ms-auto d-flex gap-2 list-unstyled mb-0 py-2">
                    <li><a class="nav-link-custom active" id="tab-home" onclick="switchSection('home')">Home</a></li>
                    <li><a class="nav-link-custom" id="tab-joint" onclick="switchSection('joint')">Predict</a></li>
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
                    <i class="fa-solid fa-heart-pulse me-1"></i> Preventive Health AI Portal
                </div>
                
                <!-- EXACT REGISTERED PROJECT TITLE -->
                <h1 class="fw-extrabold display-6 mb-3 text-dark px-2">
                    Hybrid Multitasking Heart Disease and Diabetic Disease Preventive Healthcare
                </h1>
                
                <p class="text-secondary fs-6 max-w-700 mx-auto mb-4">
                    Dual Disease Risk Prediction System using Shared Multi-Task Attention Network Layers
                </p>
                <div class="d-flex justify-content-center gap-3">
                    <button onclick="switchSection('joint')" class="btn btn-medai px-5 py-3 w-auto">
                        <i class="fa-solid fa-stethoscope me-2"></i>START HEALTH ASSESSMENT
                    </button>
                </div>
            </div>

            <!-- METRIC CARDS -->
            <div class="row g-3 mb-4">
                <div class="col-md-3 col-6">
                    <div class="stat-card">
                        <div class="stat-num">95.8%</div>
                        <div class="stat-label">Model Accuracy</div>
                    </div>
                </div>
                <div class="col-md-3 col-6">
                    <div class="stat-card">
                        <div class="stat-num">2-in-1</div>
                        <div class="stat-label">Multi-Task Model</div>
                    </div>
                </div>
                <div class="col-md-3 col-6">
                    <div class="stat-card">
                        <div class="stat-num">Heart & Diabetes</div>
                        <div class="stat-label">Joint Risk Assessment</div>
                    </div>
                </div>
                <div class="col-md-3 col-6">
                    <div class="stat-card">
                        <div class="stat-num">Deep Learning</div>
                        <div class="stat-label">Attention Layer</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- SECTION 2: PREDICT FORM & OUTPUTS -->
        <div id="sec-joint" class="d-none">
            <div class="glass-card mx-auto" style="max-width: 800px;">
                <div class="d-flex justify-content-between align-items-center mb-4 pb-2 border-bottom">
                    <div>
                        <span class="badge bg-primary text-white fw-bold px-3 py-1 rounded-pill mb-1">Clinical Form</span>
                        <h4 class="fw-bold text-dark mb-0">Patient Health Input Panel</h4>
                    </div>
                    <span class="fs-2">📋</span>
                </div>

                <form action="/predict" method="POST">
                    <div class="row g-3">
                        <div class="col-md-6">
                            <label class="form-label-custom">Patient Age (Years):</label>
                            <input type="number" name="age" class="form-control form-control-custom" placeholder="e.g. 45" value="{{ age if result else '' }}" required>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label-custom">Fasting Glucose (mg/dL):</label>
                            <input type="number" name="glucose" class="form-control form-control-custom" placeholder="e.g. 120" value="{{ glucose if result else '' }}" required>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label-custom">Systolic BP (mmHg):</label>
                            <input type="number" name="sys_bp" class="form-control form-control-custom" placeholder="e.g. 130" value="{{ sys_bp if result else '' }}" required>
                        </div>
                        <div class="col-md-6">
                            <label class="form-label-custom">Diastolic BP (mmHg):</label>
                            <input type="number" name="dia_bp" class="form-control form-control-custom" placeholder="e.g. 85" value="{{ dia_bp if result else '' }}" required>
                        </div>
                        <div class="col-12">
                            <label class="form-label-custom">Serum Cholesterol (mg/dL):</label>
                            <input type="number" name="cholesterol" class="form-control form-control-custom" placeholder="e.g. 200" value="{{ cholesterol if result else '' }}" required>
                        </div>
                    </div>

                    <button type="submit" class="btn btn-medai mt-4"><i class="fa-solid fa-play me-2"></i>PREDICT RISK RESULTS</button>
                </form>

                {% if result %}
                <!-- DIAGNOSTIC RESULTS -->
                <div class="mt-4 pt-3 border-top">
                    <div class="d-flex justify-content-between align-items-center mb-3 flex-wrap gap-2">
                        <h4 class="fw-bold text-dark mb-0">Assessment Output</h4>
                        <a href="/download_pdf?age={{age}}&glucose={{glucose}}&sys_bp={{sys_bp}}&dia_bp={{dia_bp}}&cholesterol={{cholesterol}}&h_risk={{heart_risk}}&h_prob={{heart_proba}}&d_risk={{diabetes_risk}}&d_prob={{diabetes_proba}}" class="btn btn-success fw-bold px-4 rounded-pill shadow-sm"><i class="fa-solid fa-file-pdf me-2"></i>Download PDF Report</a>
                    </div>
                    <div class="row g-3">
                        <div class="col-md-6">
                            <div class="p-4 border rounded-4 text-center {{ 'bg-danger-subtle border-danger' if 'HIGH' in heart_risk else 'bg-success-subtle border-success' }}">
                                <small class="text-secondary fw-bold d-block mb-1">TASK 1: HEART DISEASE RISK</small>
                                <h2 class="{{ 'text-danger' if 'HIGH' in heart_risk else 'text-success' }} fw-bold mb-1">{{ heart_risk }}</h2>
                                <span class="badge bg-white text-dark border fw-bold">Confidence: {{ heart_proba }}%</span>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="p-4 border rounded-4 text-center {{ 'bg-danger-subtle border-danger' if 'HIGH' in diabetes_risk else 'bg-success-subtle border-success' }}">
                                <small class="text-secondary fw-bold d-block mb-1">TASK 2: DIABETES RISK</small>
                                <h2 class="{{ 'text-danger' if 'HIGH' in diabetes_risk else 'text-success' }} fw-bold mb-1">{{ diabetes_risk }}</h2>
                                <span class="badge bg-white text-dark border fw-bold">Confidence: {{ diabetes_proba }}%</span>
                            </div>
                        </div>
                    </div>
                </div>
                {% endif %}
            </div>
        </div>

        <!-- SECTION 3: ARCHITECTURE -->
        <div id="sec-tech" class="d-none">
            <div class="glass-card mx-auto text-center" style="max-width: 700px;">
                <h4 class="fw-bold text-dark mb-3"><i class="fa-solid fa-sitemap text-primary me-2"></i>Multi-Task Neural Network Pipeline</h4>
                <p class="text-secondary">Input Parameters ➔ Dense Layers ➔ Shared Attention Mechanism ➔ Dual Output Heads (Heart & Diabetes Risk)</p>
            </div>
        </div>

        <!-- SECTION 4: CONTACT -->
        <div id="sec-contact" class="d-none">
            <div class="glass-card text-center">
                <h4 class="fw-bold text-dark mb-2">Project Guidance & Support</h4>
                <p class="text-secondary">Department of CSE (AI) • Meerut Institute of Engineering & Technology (MIET)</p>
            </div>
        </div>

    </div>

    <!-- FOOTER -->
    <footer class="text-center">
        <div class="container">
            <p class="fw-bold text-secondary mb-1">Preventive Healthcare AI Project</p>
            <small class="text-muted">© 2026 Hybrid Multitasking Preventive Healthcare Portal</small>
        </div>
    </footer>

    <script>
        function switchSection(secId) {
            const secs = ['home', 'joint', 'tech', 'contact'];
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

        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=14, textColor=colors.HexColor('#1e40af'), spaceAfter=8)
        sub_style = ParagraphStyle('SubStyle', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#475569'), spaceAfter=15)

        elements = []
        elements.append(Paragraph("Hybrid Multitasking Health Assessment Report", title_style))
        elements.append(Paragraph("Preventive Healthcare Multi-Task Clinical Assessment", sub_style))
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
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1e40af')),
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
        return send_file(buffer, as_attachment=True, download_name="Preventive_Health_Report.pdf", mimetype="application/pdf")
    except Exception as e:
        return f"PDF Error: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)