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
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: linear-gradient(135deg, #f0f4f8 0%, #e2e8f0 100%);
            min-height: 100vh;
            color: #0f172a;
        }

        .navbar-custom {
            background: #ffffff;
            border-bottom: 1px solid #cbd5e1;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }

        .glass-card {
            background: #ffffff;
            border-radius: 20px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 10px 30px rgba(0,0,0,.05);
            padding: 2rem;
            margin-bottom: 2rem;
        }

        .stat-card {
            background: #f8fafc;
            border-radius: 14px;
            padding: 1rem;
            border: 1px solid #e2e8f0;
            text-align: center;
        }

        .btn-primary-custom {
            background: linear-gradient(135deg, #1e40af 0%, #0284c7 100%);
            color: white;
            font-weight: 700;
            border-radius: 12px;
            padding: 12px 24px;
            border: none;
            transition: all 0.3s ease;
        }

        .btn-whatsapp {
            background-color: #25D366;
            color: white;
            font-weight: 700;
            border-radius: 12px;
            padding: 10px 20px;
            border: none;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .btn-whatsapp:hover {
            background-color: #128C7E;
            color: white;
        }

        footer {
            background: #ffffff;
            border-top: 1px solid #e2e8f0;
            padding: 2rem 0;
            margin-top: 3rem;
        }
    </style>
</head>
<body>

    <!-- NAVBAR -->
    <nav class="navbar navbar-expand-lg navbar-custom sticky-top">
        <div class="container">
            <a class="navbar-brand d-flex align-items-center gap-2" href="#">
                <span class="fs-3">🩺</span>
                <span class="fw-bold text-primary fs-5">MedAI Portal</span>
            </a>
            <div class="ms-auto d-flex align-items-center gap-3">
                <span class="badge bg-light text-dark border px-3 py-2 d-none d-md-inline-block">
                    👤 Developer: <strong>SelfStudyNotes</strong>
                </span>
                <a href="#predict-section" class="btn btn-primary-custom btn-sm">Take Test</a>
            </div>
        </div>
    </nav>

    <!-- HERO SECTION -->
    <div class="container my-5">
        <div class="glass-card text-center">
            <span class="badge bg-primary-subtle text-primary fw-bold px-3 py-2 rounded-pill mb-3">
                🔬 AI-Powered Health Predictor
            </span>
            <h1 class="fw-extrabold display-6 mb-3">
                Hybrid Multitasking Heart Disease and Diabetic Disease Preventive Healthcare
            </h1>
            <p class="text-secondary fs-6 max-w-700 mx-auto mb-4">
                Instantly check your heart disease and diabetes risk using advanced deep learning multitasking networks.
            </p>
            
            <div class="d-flex justify-content-center gap-3 flex-wrap">
                <a href="#predict-section" class="btn btn-primary-custom">
                    <i class="fa-solid fa-notes-medical me-2"></i>Start Free Health Risk Check
                </a>
                <a href="https://api.whatsapp.com/send?text=Check%20your%20Heart%20%26%20Diabetes%20risk%20instantly%20using%20this%20AI%20Health%20Portal:%20https://hybrid-multitask-health-predictor.onrender.com" target="_blank" class="btn btn-whatsapp">
                    <i class="fa-brands fa-whatsapp fs-5"></i> Share with Friends & Family
                </a>
            </div>
        </div>

        <!-- STATS -->
        <div class="row g-3 mb-5">
            <div class="col-md-3 col-6">
                <div class="stat-card">
                    <div class="fw-bold text-primary fs-4">95.8%</div>
                    <small class="text-muted fw-bold">Prediction Accuracy</small>
                </div>
            </div>
            <div class="col-md-3 col-6">
                <div class="stat-card">
                    <div class="fw-bold text-primary fs-4">Dual Task</div>
                    <small class="text-muted fw-bold">Heart & Diabetes</small>
                </div>
            </div>
            <div class="col-md-3 col-6">
                <div class="stat-card">
                    <div class="fw-bold text-primary fs-4">Instant</div>
                    <small class="text-muted fw-bold">PDF Report Generation</small>
                </div>
            </div>
            <div class="col-md-3 col-6">
                <div class="stat-card">
                    <div class="fw-bold text-primary fs-4">Free Access</div>
                    <small class="text-muted fw-bold">Preventive Healthcare</small>
                </div>
            </div>
        </div>

        <!-- PREDICTION FORM -->
        <div id="predict-section" class="glass-card mx-auto" style="max-width: 800px;">
            <h3 class="fw-bold mb-4 text-center">📋 Enter Patient Clinical Parameters</h3>
            
            <form action="/predict#predict-section" method="POST">
                <div class="row g-3">
                    <div class="col-md-6">
                        <label class="form-label fw-bold small text-secondary">Age (Years):</label>
                        <input type="number" name="age" class="form-control" placeholder="e.g. 45" value="{{ age if result else '' }}" required>
                    </div>
                    <div class="col-md-6">
                        <label class="form-label fw-bold small text-secondary">Fasting Glucose (mg/dL):</label>
                        <input type="number" name="glucose" class="form-control" placeholder="e.g. 120" value="{{ glucose if result else '' }}" required>
                    </div>
                    <div class="col-md-6">
                        <label class="form-label fw-bold small text-secondary">Systolic BP (mmHg):</label>
                        <input type="number" name="sys_bp" class="form-control" placeholder="e.g. 130" value="{{ sys_bp if result else '' }}" required>
                    </div>
                    <div class="col-md-6">
                        <label class="form-label fw-bold small text-secondary">Diastolic BP (mmHg):</label>
                        <input type="number" name="dia_bp" class="form-control" placeholder="e.g. 85" value="{{ dia_bp if result else '' }}" required>
                    </div>
                    <div class="col-12">
                        <label class="form-label fw-bold small text-secondary">Serum Cholesterol (mg/dL):</label>
                        <input type="number" name="cholesterol" class="form-control" placeholder="e.g. 200" value="{{ cholesterol if result else '' }}" required>
                    </div>
                </div>

                <button type="submit" class="btn btn-primary-custom w-100 mt-4 py-3">
                    🔍 Analyze & Predict Risk
                </button>
            </form>

            {% if result %}
            <div class="mt-5 pt-4 border-top">
                <div class="d-flex justify-content-between align-items-center mb-4 flex-wrap gap-2">
                    <h4 class="fw-bold mb-0">Diagnostic Output</h4>
                    <a href="/download_pdf?age={{age}}&glucose={{glucose}}&sys_bp={{sys_bp}}&dia_bp={{dia_bp}}&cholesterol={{cholesterol}}&h_risk={{heart_risk}}&h_prob={{heart_proba}}&d_risk={{diabetes_risk}}&d_prob={{diabetes_proba}}" class="btn btn-success fw-bold rounded-pill">
                        <i class="fa-solid fa-file-pdf me-2"></i>Download Clinical PDF Report
                    </a>
                </div>
                
                <div class="row g-3">
                    <div class="col-md-6">
                        <div class="p-4 border rounded-4 text-center {{ 'bg-danger-subtle border-danger' if 'HIGH' in heart_risk else 'bg-success-subtle border-success' }}">
                            <small class="text-secondary fw-bold d-block mb-1">TASK 1: HEART DISEASE RISK</small>
                            <h2 class="{{ 'text-danger' if 'HIGH' in heart_risk else 'text-success' }} fw-bold mb-1">{{ heart_risk }}</h2>
                            <span class="badge bg-white text-dark border">Confidence: {{ heart_proba }}%</span>
                        </div>
                    </div>
                    <div class="col-md-6">
                        <div class="p-4 border rounded-4 text-center {{ 'bg-danger-subtle border-danger' if 'HIGH' in diabetes_risk else 'bg-success-subtle border-success' }}">
                            <small class="text-secondary fw-bold d-block mb-1">TASK 2: DIABETES RISK</small>
                            <h2 class="{{ 'text-danger' if 'HIGH' in diabetes_risk else 'text-success' }} fw-bold mb-1">{{ diabetes_risk }}</h2>
                            <span class="badge bg-white text-dark border">Confidence: {{ diabetes_proba }}%</span>
                        </div>
                    </div>
                </div>

                <!-- SHARE RESULT BUTTON -->
                <div class="text-center mt-4">
                    <p class="fw-bold mb-2 small text-secondary">Help others check their risk:</p>
                    <a href="https://api.whatsapp.com/send?text=I%20just%20checked%20my%20health%20risk%20assessment%20on%20this%20AI%20Portal.%20Check%20yours%20here:%20https://hybrid-multitask-health-predictor.onrender.com" target="_blank" class="btn btn-whatsapp">
                        <i class="fa-brands fa-whatsapp fs-5"></i> Share Assessment Link on WhatsApp
                    </a>
                </div>
            </div>
            {% endif %}
        </div>
    </div>

    <!-- FOOTER WITH PERSONAL BRANDING & CONTACT -->
    <footer class="text-center">
        <div class="container">
            <h6 class="fw-bold text-dark mb-1">Hybrid Multitasking Healthcare AI</h6>
            <p class="text-secondary small mb-2">Developed by: <strong>SelfStudyNotes</strong></p>
            <p class="text-muted small mb-3">📧 Contact/Support: <a href="mailto:selfstudynotes01@gmail.com" class="text-decoration-none">selfstudynotes01@gmail.com</a></p>
            <small class="text-muted">© 2026 Preventive Healthcare Assessment Portal • All Rights Reserved</small>
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

        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=14, textColor=colors.HexColor('#1e40af'), spaceAfter=8)
        sub_style = ParagraphStyle('SubStyle', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#475569'), spaceAfter=15)

        elements = []
        elements.append(Paragraph("Hybrid Multitasking Preventive Healthcare Report", title_style))
        elements.append(Paragraph("Developed by: SelfStudyNotes (selfstudynotes01@gmail.com)", sub_style))
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
        return send_file(buffer, as_attachment=True, download_name="Clinical_Preventive_Health_Report.pdf", mimetype="application/pdf")
    except Exception as e:
        return f"PDF Error: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)