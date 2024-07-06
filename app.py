import cv2
import numpy as np
import face_recognition
import streamlit as st
from PIL import Image

# Charger les signatures
signatures_class = np.load('FaceSignatures_db.npy')
X = signatures_class[:, 0:-1].astype('float')
Y = signatures_class[:, -1]

# Barre latérale pour télécharger une image
st.title("Facial Recognition")
img = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png", "jfif"])

if img is not None:
    img = Image.open(img)
    img_numpy = np.array(img)
    resized_image = cv2.resize(img_numpy, (0, 0), None, 0.25, 0.25)

    # Convertir la couleur
    resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Détecter les visages
    facesCurrent = face_recognition.face_locations(resized_image)

    # Si aucun visage n'est détecté
    if not facesCurrent:
        st.warning("Aucun visage détecté dans l'image téléchargée. Veuillez essayer avec une autre image.")
    else:
        # Extraction des caractéristiques faciales
        encodesCurrent = face_recognition.face_encodings(resized_image, facesCurrent)

        # Comparaison des caractéristiques faciales
        similarity_threshold = 0.6  # seuil de similarité pour afficher les visages
        found_faces = []

        for encodeFace, faceLoc in zip(encodesCurrent, facesCurrent):
            matches = face_recognition.compare_faces(X, encodeFace, tolerance=similarity_threshold)
            faceDis = face_recognition.face_distance(X, encodeFace)

            # Trouver les indices des visages correspondants
            match_indices = np.where(matches)[0]

            if match_indices.size > 0:
                for index in match_indices:
                    name = Y[index].upper()
                    pht = f'./images/{name}.jpg'
                    found_faces.append((name, pht))

        # Afficher les visages correspondants côte à côte
        if found_faces:
            num_faces = len(found_faces)
            cols = st.columns(num_faces)  # Crée des colonnes pour chaque visage

            # Affiche chaque visage dans une colonne
            for i, (name, pht) in enumerate(found_faces):
                with cols[i]:
                    st.write(f"Nom : {name}")
                    st.image(pht, use_column_width=True)
        else:
            st.warning("Visage non reconnu. Aucun correspondant trouvé dans la base de données.")
