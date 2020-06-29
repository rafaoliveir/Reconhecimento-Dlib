import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np


#detectar face
detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
reconhecimentoFacial = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

indice = {}
idx = 0
descritoresFaciais = None

for arquivo in glob.glob(os.path.join("dark", "*.gif")):
    imagem = cv2.imread(arquivo)  #leitura das imagens
    facesDetectadas = detectorFace(imagem, 1)
    numeroFacesDetectadas = len(facesDetectadas)
    # print(numeroFacesDetectadas)
    if numeroFacesDetectadas > 1:
        print("Há mais de uma face na imagem {}".format(arquivo))
        exit(0)
    elif numeroFacesDetectadas < 1:
        print("Nenhuma face encontrada {}".format(arquivo))
        exit(0)

    for face in facesDetectadas:
        pontosFaciais = detectorPontos(imagem, face) #extrair os pontos faciais
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)
        # print(format(arquivo))
        # print(len(descritorFacial))
        # print(descritorFacial)
        listaDescritorFacial = [df for df in descritorFacial]
        #print(listaDescritorFacial)

        npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
        #print(npArrayDescritorFacial)

        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]
        #print(npArrayDescritorFacial)

        if descritoresFaciais is None:
            descritoresFaciais = npArrayDescritorFacial
        else:
            descritoresFaciais = np.concatenate((descritoresFaciais, npArrayDescritorFacial), axis=0)

        indice[idx] = arquivo
        idx += 1


    #cv2.imshow("Treinamento", imagem)
    #cv2.waitKey(0)

#print("Tamanho: {} Formato: {}".format(len(descritoresFaciais), descritoresFaciais.shape))
#print(descritoresFaciais)
#print(indice)

np.save("jonas_rn.npy", descritoresFaciais)
with open("indices_rn.pickle", 'wb') as f:
            cPickle.dump(indice, f)
#cv2.destroyAllWindows()