import dlib
import cv2
import numpy as np

#visualizar pontos detectados
def imprimeNumeros(imagem, pontosFaciais):
    for p in pontosFaciais.parts():
        cv2.circle(imagem, (p.x, p.y), 2, (0,255,0), 2) #raio, cor e borda da marcação

def imprimeNumeros(imagem, pontosFaciais):
        for i, p in enumerate(pontosFaciais.parts()):
                cv2.putText(imagem, str(i), (p.x, p.y), fonte, .55, (0, 0, 255), 1) #vai dizer onde está os pontos

def imprimeLinhas(imagem, pontosFaciais):
    p68 = [[0, 16, False],  # linha do queixo
           [17, 21, False],  # sombrancelha direita
           [22, 26, False],  # sombancelha esquerda
           [27, 30, False],  # ponte nasal
           [30, 35, True],  # nariz inferior
           [36, 41, True],  # olho esquerdo
           [42, 47, True],  # olho direito
           [48, 59, True],  # labio externo
           [60, 67, True]]  # labio interno
    for k in range(0, len(p68)):
        pontos = []
        for i in range(p68[k][0], p68[k][1] + 1):
            ponto = [pontosFaciais.part(i).x, pontosFaciais.part(i).y]
            pontos.append(ponto)
        pontos = np.array(pontos, dtype=np.int32)
        cv2.polylines(imagem, [pontos], p68[k][2], (255, 0, 0), 2)


fonte=cv2.FONT_HERSHEY_COMPLEX_SMALL
imagem = cv2.imread("dark/jonas2.gif")
detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
facesDetectadas= detectorFace(imagem, 2)
for face in facesDetectadas:
    pontos = detectorPontos(imagem, face)
    print(pontos.parts())
    print(len(pontos.parts()))
    #imprimePontos(imagem, pontos)
    #imprimeNumeros(imagem, pontos)
    imprimeLinhas(imagem, pontos)

cv2.imshow("Pontos Faciais", imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()