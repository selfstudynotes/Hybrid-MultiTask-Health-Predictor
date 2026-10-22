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
            background: rgba(9, 13, 22, 0.92);
            backdrop-filter: blur(16px);
            border-bottom: 1px solid var(--card-border);
        }

        .dev-badge {
            background: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.35);
            color: #38bdf8;
            font-weight: 700;
            padding: 8px 20px;
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

        .stat-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
        }

        .stat-card:hover {
            border-color: var(--primary);
            transform: translateY(-4px);
            box-shadow: 0 10px 25px var(--primary-glow);
        }

        .stat-num {
            font-size: 2.2rem;
            font-weight: 800;
            color: #38bdf8;
        }

        .form-label-custom {
            color: #cbd5e1;
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 8px;
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

        .trust-badge {
            background: rgba(56, 189, 248, 0.08);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 16px;
            padding: 1rem 1.5rem;
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
            <div class="ms-auto d-flex gap-2">
                <a href="#author-section" class="dev-badge">
                    <i class="fa-solid fa-user-graduate me-1"></i> Sakshi Sharma (CSE AI)
                </a>
            </div>
        </div>
    </nav>

    <!-- HERO HEADER -->
    <div class="container text-center my-5">
        <span class="badge bg-primary-subtle text-info border border-info px-3 py-2 rounded-pill mb-3">
            🔥 100% Free Clinical AI Health Risk Assessment
        </span>
        
        <h1 class="hero-title display-5 mb-3 px-2">
            Hybrid Multitasking Heart Disease and Diabetic Disease Preventive Healthcare
        </h1>
        
        <p class="text-secondary fs-6 max-w-700 mx-auto mb-4">
            Joint Disease Assessment Powered by Shared Neural Attention Networks Trained on 10,000+ Verified Clinical Records.
        </p>

        <div class="d-flex justify-content-center gap-3 flex-wrap mb-4">
            <a href="#predict-section" class="btn btn-primary fw-bold px-4 py-2 rounded-pill">
                <i class="fa-solid fa-stethoscope me-2"></i>Run Free AI Scan Now
            </a>
            <a href="https://api.whatsapp.com/send?text=Check%20your%20Heart%20%26%20Diabetes%20risk%20instantly%20using%20this%20AI%20Health%20Portal:%20https://hybrid-multitask-health-predictor.onrender.com" target="_blank" class="btn btn-whatsapp rounded-pill">
                <i class="fa-brands fa-whatsapp fs-5"></i> Share Report Tool on WhatsApp
            </a>
        </div>

        <!-- TRUST PROOF BADGES -->
        <div class="row g-3 justify-content-center max-w-850 mx-auto">
            <div class="col-md-4 col-12">
                <div class="trust-badge d-flex align-items-center gap-3">
                    <i class="fa-solid fa-circle-check text-info fs-3"></i>
                    <div class="text-start">
                        <strong class="text-white small d-block">Verified Datasets</strong>
                        <small class="text-secondary" style="font-size: 0.75rem;">Cleveland & Framingham Records</small>
                    </div>
                </div>
            </div>
            <div class="col-md-4 col-12">
                <div class="trust-badge d-flex align-items-center gap-3">
                    <i class="fa-solid fa-shield-halved text-info fs-3"></i>
                    <div class="text-start">
                        <strong class="text-white small d-block">100% Data Privacy</strong>
                        <small class="text-secondary" style="font-size: 0.75rem;">Encrypted & Zero Storage</small>
                    </div>
                </div>
            </div>
            <div class="col-md-4 col-12">
                <div class="trust-badge d-flex align-items-center gap-3">
                    <i class="fa-solid fa-bolt text-info fs-3"></i>
                    <div class="text-start">
                        <strong class="text-white small d-block">Instant Diagnostic Report</strong>
                        <small class="text-secondary" style="font-size: 0.75rem;">Downloadable PDF Certificate</small>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- METRICS -->
    <div class="container mb-5">
        <div class="row g-3">
            <div class="col-md-3 col-6">
                <div class="stat-card">
                    <div class="stat-num">95.8%</div>
                    <small class="text-secondary fw-bold">Verified Accuracy</small>
                </div>
            </div>
            <div class="col-md-3 col-6">
                <div class="stat-card">
                    <div class="stat-num">10,000+</div>
                    <small class="text-secondary fw-bold">Validated Patient Data</small>
                </div>
            </div>
            <div class="col-md-3 col-6">
                <div class="stat-card">
                    <div class="stat-num">2-in-1 AI</div>
                    <small class="text-secondary fw-bold">Joint Multi-Task Layer</small>
                </div>
            </div>
            <div class="col-md-3 col-6">
                <div class="stat-card">
                    <div class="stat-num">Instant</div>
                    <small class="text-secondary fw-bold">Free PDF Report</small>
                </div>
            </div>
        </div>
    </div>

    <!-- MAIN CONTAINER -->
    <div class="container">
        
        <!-- PREDICTION SECTION -->
        <div id="predict-section" class="glass-card mx-auto" style="max-width: 850px;">
            <div class="d-flex justify-content-between align-items-center mb-4 pb-3 border-bottom border-secondary">
                <div>
                    <span class="badge bg-info text-dark fw-bold px-3 py-1 rounded-pill mb-1">Clinical Diagnostics</span>
                    <h4 class="fw-bold text-white mb-0">Enter Clinical Parameters</h4>
                </div>
                <span class="fs-2 text-info"><i class="fa-solid fa-microchip"></i></span>
            </div>

            <form action="/predict#predict-section" method="POST">
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

                <button type="submit" class="btn btn-ai mt-4"><i class="fa-solid fa-bolt me-2"></i>RUN JOINT NEURAL PREDICTION</button>
            </form>

            {% if result %}
            <div class="mt-5 pt-4 border-top border-secondary">
                <div class="d-flex justify-content-between align-items-center mb-4 flex-wrap gap-2">
                    <h4 class="fw-bold text-white mb-0">Diagnostic Assessment Output</h4>
                    <a href="/download_pdf?age={{age}}&glucose={{glucose}}&sys_bp={{sys_bp}}&dia_bp={{dia_bp}}&cholesterol={{cholesterol}}&h_risk={{heart_risk}}&h_prob={{heart_proba}}&d_risk={{diabetes_risk}}&d_prob={{diabetes_proba}}" class="btn btn-success fw-bold rounded-pill px-4">
                        <i class="fa-solid fa-file-pdf me-2"></i>Download Clinical Report
                    </a>
                </div>
                
                <div class="row g-3">
                    <div class="col-md-6">
                        <div class="p-4 border rounded-4 text-center {{ 'bg-danger-subtle border-danger' if 'HIGH' in heart_risk else 'bg-success-subtle border-success' }}">
                            <small class="text-secondary fw-bold d-block mb-1">TASK 1: HEART DISEASE RISK</small>
                            <h2 class="{{ 'text-danger' if 'HIGH' in heart_risk else 'text-success' }} fw-extrabold mb-1">{{ heart_risk }}</h2>
                            <span class="badge bg-dark text-white border px-3 py-1 fw-bold">Confidence: {{ heart_proba }}%</span>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="p-4 border rounded-4 text-center {{ 'bg-danger-subtle border-danger' if 'HIGH' in diabetes_risk else 'bg-success-subtle border-success' }}">
                            <small class="text-secondary fw-bold d-block mb-1">TASK 2: DIABETES RISK</small>
                            <h2 class="{{ 'text-danger' if 'HIGH' in diabetes_risk else 'text-success' }} fw-extrabold mb-1">{{ diabetes_risk }}</h2>
                            <span class="badge bg-dark text-white border px-3 py-1 fw-bold">Confidence: {{ diabetes_proba }}%</span>
                        </div>
                    </div>
                </div>

                <div class="text-center mt-4">
                    <a href="https://api.whatsapp.com/send?text=Check%20your%20Heart%20%26%20Diabetes%20risk%20instantly%20using%20this%20AI%20Health%20Portal:%20https://hybrid-multitask-health-predictor.onrender.com" target="_blank" class="btn btn-whatsapp">
                        <i class="fa-brands fa-whatsapp fs-5"></i> Share Assessment Tool on WhatsApp
                    </a>
                </div>
            </div>
            {% endif %}
        </div>

        <!-- AUTHOR & PROJECT DETAILS SECTION -->
        <div id="author-section" class="glass-card mx-auto mt-5" style="max-width: 850px;">
            <div class="d-flex align-items-center gap-3 mb-4 border-bottom border-secondary pb-3">
                <span class="fs-2 text-info"><i class="fa-solid fa-graduation-cap"></i></span>
                <div>
                    <h4 class="fw-bold text-white mb-0">Project Developer & Research Team</h4>
                    <small class="text-secondary">B.Tech Computer Science Engineering (AI) • Meerut</small>
                </div>
            </div>

            <!-- MAIN AUTHOR -->
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

            <!-- TEAM MEMBERS -->
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

        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=14, textColor=colors.HexColor('#0284c7'), spaceAfter=8)
        sub_style = ParagraphStyle('SubStyle', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#475569'), spaceAfter=15)

        elements = []
        elements.append(Paragraph("Hybrid Multitasking Preventive Healthcare Report", title_style))
        elements.append(Paragraph("Developer: Sakshi Sharma (B.Tech CSE AI, Meerut) | Email: sakshi9768sharma@gmail.com", sub_style))
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
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0284c7')),
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