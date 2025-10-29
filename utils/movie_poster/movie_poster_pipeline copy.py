from fastai.vision.all import *
import pandas as pd
import re
import torch
import os, torch

def read_movies_genre(path):
    data = pd.read_csv(
        path,
        delimiter= ";",
        encoding='latin1',       # o prueba 'cp1252' si falla
        on_bad_lines='skip',     # salta filas corruptas
        engine='python'          # lector más flexible
    )
    data.dropna(inplace=True)

    return data

def get_float_labels(file_name):
    return float(re.search('\d.\d',str(file_name)).group())
def get_score_labels(file_name):
    return re.search('\d.\d',str(file_name)).group()

def pred_regresion(path_img):
    data_reg = DataBlock(
        blocks=(ImageBlock, RegressionBlock),
        get_items=get_image_files,
        get_y=get_float_labels,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        item_tfms=Resize([300,180]),
        batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)]
    )
    # 👇 sin subprocesos
    dls = data_reg.dataloaders(path_img, bs=16, num_workers=0, shuffle=True, persistent_workers=False)
    return dls

def pred_clasificacion(path_img):
    data_class = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        get_y=get_score_labels,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        item_tfms=Resize([300,180]),
        batch_tfms=[*aug_transforms(), Normalize.from_stats(*imagenet_stats)]
    )
    dls_class = data_class.dataloaders(path_img, bs=16, num_workers=0, shuffle=True, persistent_workers=False)
    return dls_class

def leaner_regresion(dls):

    class L1LossFlat(nn.L1Loss):
        "Mean Absolute Error Loss (FastAI v2 compatible)"
        def forward(self, input: Tensor, target: Tensor) -> torch.Tensor:
            return super().forward(input.view(-1), target.view(-1))
        
    # ---Learner para REGRESIÓN ---
    learn_reg = cnn_learner(
        dls=dls,              # tus DataLoaders de regresión
        arch=resnet50,            # arquitectura base
        loss_func=L1LossFlat(),   # pérdida MAE
        metrics=mae               # métrica de error absoluto medio
    )

    print("Inicio Entrenamiento regresion")
    learn_reg.fit_one_cycle(5)
    print("Fin Entrenamiento regresion")

    return learn_reg

def leaner_clases(dls_class):

    # ---Learner para CLASIFICACIÓN ---
    learn_class = cnn_learner(
        dls=dls_class,            # tus DataLoaders de clasificación
        arch=resnet50,            # misma arquitectura
        loss_func=CrossEntropyLossFlat(),  # pérdida de clasificación
        metrics=accuracy          # métrica de precisión
    )

    print("Inicio Entrenamiento clasificacion")
    learn_class.fit_one_cycle(5)
    print("Fin Entrenamiento clasificacion")

    return learn_class

def carga_imagenes(path,learn_reg,learn_class):

    lista_de_imagenes = [
        "test1.jpg",
        "test2.jpg",
        "test3.jpg",
        "test4.jpg",
        "test5.jpg",
        "test6.jpg",
        "test7.jpg",
        "test8.jpg",
        "test9.jpg",
        "test10.jpg",
    ]

    rows = []
    for img_path in lista_de_imagenes:
        img = PILImage.create(img_path)
        reg_pred = learn_reg.predict(img)[0]
        cls_pred = learn_class.predict(img)[0]
        rows.append([img_path, float(reg_pred), str(cls_pred)])

    results = pd.DataFrame(rows, columns=['Imagen', 'Pred_IMDB_Regression', 'Pred_IMDB_Classification'])
    results.to_excel(path + r'\predicciones_peliculas.xlsx', index=False)

    return results


# def leaner(dls,dls_class,path):

#     class L1LossFlat(nn.L1Loss):
#         "Mean Absolute Error Loss (FastAI v2 compatible)"
#         def forward(self, input: Tensor, target: Tensor) -> torch.Tensor:
#             return super().forward(input.view(-1), target.view(-1))
        
#     # ---Learner para REGRESIÓN ---
#     learn_reg = cnn_learner(
#         dls=dls,              # tus DataLoaders de regresión
#         arch=resnet50,            # arquitectura base
#         loss_func=L1LossFlat(),   # pérdida MAE
#         metrics=mae               # métrica de error absoluto medio
#     )

#     # ---Learner para CLASIFICACIÓN ---
#     learn_class = cnn_learner(
#         dls=dls_class,            # tus DataLoaders de clasificación
#         arch=resnet50,            # misma arquitectura
#         loss_func=CrossEntropyLossFlat(),  # pérdida de clasificación
#         metrics=accuracy          # métrica de precisión
#     )
#     print("Inicio Entrenamiento regresion")
#     learn_reg.fit_one_cycle(5)
#     print("Fin Entrenamiento regresion")

#     print("Inicio Entrenamiento clasificacion")
#     learn_class.fit_one_cycle(5)
#     print("Fin Entrenamiento clasificacion")

#     lista_de_imagenes = [
#         "test1.jpg",
#         "test2.jpg",
#         "test3.jpg",
#         "test4.jpg",
#         "test5.jpg",
#         "test6.jpg",
#         "test7.jpg",
#         "test8.jpg",
#         "test9.jpg",
#         "test10.jpg",
#     ]

#     rows = []
#     for img_path in lista_de_imagenes:
#         img = PILImage.create(img_path)
#         reg_pred = learn_reg.predict(img)[0]
#         cls_pred = learn_class.predict(img)[0]
#         rows.append([img_path, float(reg_pred), str(cls_pred)])

#     results = pd.DataFrame(rows, columns=['Imagen', 'Pred_IMDB_Regression', 'Pred_IMDB_Classification'])
#     results.to_excel(path + r'\predicciones_peliculas.xlsx', index=False)

#     return results