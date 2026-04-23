import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import winsound
import os

# --- 1. الإعدادات والبيانات الشخصية ---
print("--- 🩺 Welcome to Hasna's Ultra Gait Analysis System ---")
p_name = input("Enter Patient Name: ")
p_age = input("Enter Patient Age: ") # إضافة السن
video_path = r'C:\Users\hasna fathy\Desktop\ai_test\test.mp4'
desktop = os.path.join(os.environ['USERPROFILE'], 'Desktop')

# تعريف المكتبات
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
data_log = []

def calculate_angle(a,b,c):
    a,b,c = np.array(a),np.array(b),np.array(c)
    ang = np.abs(np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0]))
    val = np.degrees(ang)
    return val if val <= 180 else 360-val

def detect_phase(k_ang, a_ang):
    if k_ang > 165 and a_ang < 110: return "Heel Strike"
    if k_ang < 140: return "Swing Phase"
    return "Stance Phase"

cap = cv2.VideoCapture(video_path)
K_T, H_T, A_T = 150, 145, 120 # عتبات التنبيه

print(f"🚀 Processing {p_name} ({p_age} years old)... Please wait.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if res.pose_landmarks:
        lm = res.pose_landmarks.landmark
        mp_drawing.draw_landmarks(image, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # حساب الزوايا Bilateral
        lk = calculate_angle([lm[23].x,lm[23].y],[lm[25].x,lm[25].y],[lm[27].x,lm[27].y])
        rk = calculate_angle([lm[24].x,lm[24].y],[lm[26].x,lm[26].y],[lm[28].x,lm[28].y])
        lh = calculate_angle([lm[11].x,lm[11].y],[lm[23].x,lm[23].y],[lm[25].x,lm[25].y])
        rh = calculate_angle([lm[12].x,lm[12].y],[lm[24].x,lm[24].y],[lm[26].x,lm[26].y])
        la = calculate_angle([lm[25].x,lm[25].y],[lm[27].x,lm[27].y],[lm[31].x,lm[31].y])
        ra = calculate_angle([lm[26].x,lm[26].y],[lm[28].x,lm[28].y],[lm[32].x,lm[32].y])

        current_phase = detect_phase(lk, la)
        data_log.append({'F':cap.get(cv2.CAP_PROP_POS_FRAMES), 'LK':lk,'RK':rk,'LH':lh,'RH':rh,'LA':la,'RA':ra, 'Phase': current_phase})

        # تنبيهات صوتية
        if lk < K_T: winsound.Beep(500, 30)

        # واجهة المستخدم (UI)
        cv2.rectangle(image, (0,0), (350, 160), (0,0,0), -1)
        cv2.putText(image, f"Patient: {p_name} | Age: {p_age}", (10, 30), 1, 1.2, (255, 255, 255), 1)
        cv2.putText(image, f"PHASE: {current_phase}", (10, 65), 1, 1.8, (0, 255, 255), 2)
        cv2.putText(image, f"KNEE L/R: {int(lk)}|{int(rk)}", (10, 105), 1, 1.3, (0, 255, 0), 1)
        cv2.putText(image, f"HIP  L/R: {int(lh)}|{int(rh)}", (10, 140), 1, 1.3, (0, 255, 0), 1)
        
    cv2.imshow('Hasna Advanced Clinical Gait', image)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# --- 2. التحليل الإحصائي والجرافات ---
df = pd.DataFrame(data_log)
# تنعيم بيانات الأنكل
df['LA'] = df['LA'].rolling(window=10, min_periods=1).mean()
df['RA'] = df['RA'].rolling(window=10, min_periods=1).mean()

mean_l = df['LK'].mean()
mean_r = df['RK'].mean()
symmetry = round(100 - (abs(mean_l - mean_r) / ((mean_l + mean_r)/2) * 100), 2)

# حفظ التقرير
report_path = os.path.join(desktop, f"{p_name}_Final_Master_Report.txt")
with open(report_path, "w") as f:
    f.write(f"GAIT ANALYSIS MASTER REPORT\n{'='*30}\n")
    f.write(f"Patient Name: {p_name}\nPatient Age: {p_age}\n")
    f.write(f"Symmetry Score: {symmetry}%\nMain Phase: {df['Phase'].mode()[0]}\n")
    f.write(f"Status: {'Excellent Symmetry' if symmetry > 90 else 'Asymmetry Detected'}\n")

# إنشاء الجرافات جنب بعض (3x2)
fig, axs = plt.subplots(3, 2, figsize=(15, 20))

def draw_bilateral(ax_pair, l_data, r_data, thresh, label):
    # Left
    ax_pair[0].plot(df['F'], l_data, color='green', label='Left')
    ax_pair[0].axhline(y=thresh, color='red', linestyle='--')
    ax_pair[0].set_title(f"{label} - Left")
    ax_pair[0].legend()
    # Right
    ax_pair[1].plot(df['F'], r_data, color='blue', label='Right')
    ax_pair[1].axhline(y=thresh, color='red', linestyle='--')
    ax_pair[1].set_title(f"{label} - Right")
    ax_pair[1].legend()

draw_bilateral(axs[0], df['LH'], df['RH'], H_T, 'Hip Angle')
draw_bilateral(axs[1], df['LK'], df['RK'], K_T, 'Knee Angle')
draw_bilateral(axs[2], df['LA'], df['RA'], A_T, 'Ankle Angle (Filtered)')

plt.tight_layout()
plt.savefig(os.path.join(desktop, f"{p_name}_Full_Grafts.png"))
plt.show(block=True) # لضمان ظهور الرسمة

print(f"✅ DONE! Symmetry: {symmetry}%")
