import cv2, numpy as np, face_recognition, os

# ImgeDB path
path = './images'

# Global variables
image_list = [] # List of images
name_List = [] # List of image names
# Gral all images from the folder
myList = os.listdir(path)
print(myList)

# Load images
for img in myList:
    if os.path.splitext(img)[1].lower() in ['.jpg', '.png', '.jpeg']: #renvoie l'extension du fichier avec un point. 
        curImg = cv2.imread(os.path.join(path, img))
#Ajoute l'img  chargé a la liste de l'img
        image_list.append(curImg)
        imgName = os.path.splitext(img)[0] #extrait nom du fichier sans extension
        name_List.append(imgName) #ajoute ce nom à la liste des noms.

# Fonction d'Extraction des Encodages Faciaux
def findEncodings(img_list, ImgName_list):

    signatures_db = []
    count = 1
    #print(ImgName_list)
    for myImg, name in zip(img_list, ImgName_list):
        img = cv2.cvtColor(myImg, cv2.COLOR_BGR2RGB)
#extrait l'encodage facial du premier visage détecté
        signature = face_recognition.face_encodings(img)[0]
#Chaque signature (encodage) est convertie en liste et le nom correspondant est ajouté.
        signature_class = signature.tolist() + [name]
#Les signatures sont ajoutées à la liste signatures_db
        signatures_db.append(signature_class)
#Affiche le pourcentage d'encodages extraits.
        print(f'{int((count/ (len(img_list)))*100)} % extracted')
        count +=1
#Les encodages faciaux sont stockés sous forme de tableau NumPy 
    signatures_db =  np.array(signatures_db)
    np.save('FaceSignatures_db.npy', signatures_db)
    print('Signature_db stored')
    
def main():
    findEncodings(image_list, name_List)

if __name__ =='__main__':
    main()
    
