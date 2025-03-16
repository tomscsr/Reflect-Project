import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import Label
import numpy as np
import threading
import time

# Initialisation de Mediapipe pour la détection faciale
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Configuration de la caméra
cap = cv2.VideoCapture(0)

# Dernière mise à jour des recommandations (évite de spammer)
last_update_time = 0

# Création de la fenêtre Tkinter (elle sera mise à jour et non recréée)
root = tk.Tk()
root.title("Recommandations Personnalisées")
root.geometry("400x300")

# Ajout des éléments dans la fenêtre Tkinter (fixe)
label_title = Label(root, text="💡 Analyse en cours...", font=("Arial", 12))
label_title.pack(pady=10)

label_info = Label(root, text="", font=("Arial", 11), justify="left")
label_info.pack()

# **Fonction pour analyser le visage (forme + teint)**
def analyze_face(frame, bbox):
    h, w, _ = frame.shape
    x, y, w_box, h_box = int(bbox.xmin * w), int(bbox.ymin * h), int(bbox.width * w), int(bbox.height * h)

    # Déterminer la **forme du visage**
    aspect_ratio = h_box / w_box
    if aspect_ratio > 1.3:
        face_shape = "Allongé"
    elif 1.1 <= aspect_ratio <= 1.3:
        face_shape = "Ovale"
    elif 0.9 <= aspect_ratio < 1.1:
        face_shape = "Rond"
    else:
        face_shape = "Carré"

    # **Analyse du ton de peau**
    roi = frame[y:y + h_box, x:x + w_box]  # Extraction du visage
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv_roi[:, :, 2])  # Calcul de la luminance

    if brightness < 80:
        skin_tone = "Foncé"
    elif brightness < 150:
        skin_tone = "Moyen"
    else:
        skin_tone = "Clair"

    return face_shape, skin_tone

# **Générer des recommandations adaptées**
def get_recommendations(face_shape, skin_tone):
    recommendations = []

    # **Fond de teint**
    if skin_tone == "Clair":
        recommendations.append("✨ Fond de teint L'Oréal True Match - Beige Rosé")
    elif skin_tone == "Moyen":
        recommendations.append("🌿 BB Cream Hydratante L'Oréal - Teinte Miel")
    else:
        recommendations.append("🌅 Fond de teint Infallible - Chocolat Profond")

    # **Soin en fonction de la forme du visage**
    if face_shape in ["Ovale", "Allongé"]:
        recommendations.append("💧 Sérum Repulpant à l'acide hyaluronique")
    elif face_shape == "Rond":
        recommendations.append("🌱 Crème Matifiante anti-brillance")
    else:
        recommendations.append("💎 Crème Éclat Vitamin C pour booster la luminosité")

    # **Barbe et grooming si applicable**
    if face_shape == "Carré":
        recommendations.append("✂️ Huile nourrissante pour barbe courte et dense")
    elif face_shape == "Allongé":
        recommendations.append("🪒 Mousse à raser peau sensible")

    return recommendations

# **Mettre à jour l'affichage Tkinter**
def update_recommendations(face_shape, skin_tone):
    recommendations = get_recommendations(face_shape, skin_tone)

    # Mise à jour de l'affichage dans la fenêtre
    label_title.config(text=f"💡 Conseils pour un visage {face_shape} et une peau {skin_tone} :")
    
    # Construire le texte des recommandations
    recommendations_text = "\n".join(recommendations)
    label_info.config(text=recommendations_text)

    # Rafraîchir Tkinter
    root.update_idletasks()

# **Boucle principale de détection du visage**
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Conversion en RGB pour Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y, w_box, h_box = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)

            # Dessiner un cadre autour du visage détecté
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

            # **Mettre à jour les recommandations toutes les 5 secondes**
            current_time = time.time()
            if current_time - last_update_time > 5:  # Attendre 5 secondes avant une nouvelle mise à jour
                last_update_time = current_time
                face_shape, skin_tone = analyze_face(frame, bboxC)
                update_recommendations(face_shape, skin_tone)

    # Effet miroir pour affichage naturel
    frame = cv2.flip(frame, 1)

    # Affichage de la caméra
    cv2.imshow("Détection & Recommandations Personnalisées", frame)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libération des ressources
cap.release()
cv2.destroyAllWindows()
