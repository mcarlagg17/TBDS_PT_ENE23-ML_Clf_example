from utils.libreries import *

#           ###
# **Funciones Auxiliares:**
#           ###

def label_encode_columns(df: pd.DataFrame,encoder = LabelEncoder()):
    '''
    Objetivo:
    ---
    Codificar las columnas categóricas.

    args.
    ---
    df: dataframe
    encoder: objeto encoder 
        

    ret.
    ---
    df: dataframe con las columnas codificadas
    
    '''

    object_cols = df.select_dtypes(include=['object']).columns
    
    for col in object_cols:
        df[col] = encoder.fit_transform(df[col])
    
    return df

def eval_metrics(y_pred: pd.Series, y_test: pd.Series, clf: bool = True,c_matrix: bool = False) -> dict or tuple:
    '''
    Objetivo:
    ---
    Evaluar el modelo supervisado utilizando las métricas correspondientes.

    args.
    ---
    y_pred: La predicción del modelo.
    y_test: El resultado de prueba real.
    clf: bool; True: si es clasificación. (por defecto)
                False: si es regresión.
    c_matrix: bool; True: obtener la matriz de confusión.
                    False: no obtener la matriz. (por defecto)

    ret.
    ---
    dict; resultados de las métricas.

    * Excepto cuando c_matrix es True y clf es True:
        dict, array; resultados de las métricas, matriz de confusión.

    '''

    if clf:
        
        clf_metrics = {
            'ACC' : accuracy_score(y_test,y_pred),
            'Precision' : precision_score(y_test,y_pred),
            'Recall' : recall_score(y_test,y_pred),
            'F1' : f1_score(y_test,y_pred),
            'ROC' : roc_auc_score(y_test,y_pred),
            'Jaccard' : jaccard_score(y_test,y_pred)
        }
        
        if c_matrix:
            confusion_mtx = confusion_matrix(y_test,y_pred)
            return clf_metrics,confusion_mtx # type: ignore
        else:
            return clf_metrics

    else:

        reg_metrics = {
            'MAE' : mean_absolute_error(y_test,y_pred),
            'MAPE' : mean_absolute_percentage_error(y_test,y_pred),
            'MSE' : mean_squared_error(y_test,y_pred),
            'R2' : r2_score(y_test,y_pred)
        }   

        return reg_metrics  

def return_categorical(dataframe: pd.DataFrame) -> pd.DataFrame:
    '''
    Objectivo: 
    ---
    Convierte las columnas de tipo "object" en tipo "categorical".

    args.
    ---
    dataframe: dataframe

    ret.
    ---
    dataframe: dataframe con las columnas codificadas
    '''
    # Obtener las columnas de tipo "object"
    columnas_object = dataframe.select_dtypes(include='object').columns

    # Convertir las columnas a tipo "categorical"
    dataframe[columnas_object] = dataframe[columnas_object].astype('category')

    # Devolver el dataframe modificado
    return dataframe


#           ###
# **Funciones Machine Learning:**
#           ###

# Plot learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Objetivo:
    ---
    Generar una gráfica simple de la curva de aprendizaje del conjunto de prueba y entrenamiento.

    args.
    ----------
    estimator: objeto que implementa los métodos "fit" y "predict"
        Un objeto de este tipo que se clona para cada validación.

    title: string
        Título para la gráfica.

    X: array-like, shape (n_samples, n_features)
        Vector de entrenamiento, donde n_samples es el número de muestras y
        n_features es el número de características.

    y: array-like, shape (n_samples) o (n_samples, n_features), opcional
        Objetivo relativo a X para clasificación o regresión;
        None para aprendizaje no supervisado.

    ylim: tuple, shape (ymin, ymax), opcional
        Define los valores mínimos y máximos de y que se mostrarán en la gráfica.

    cv: entero o generador de validación cruzada, opcional
        Si se pasa un entero, representa el número de pliegues (por defecto es 3).
        Se pueden pasar objetos específicos de validación cruzada, ver
        el módulo sklearn.cross_validation para obtener la lista de objetos posibles.

    n_jobs: entero, opcional
        Número de trabajos a ejecutar en paralelo (por defecto es 1).

    """

    plt.figure(figsize=(15,8))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
        
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores  = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes) # type: ignore
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# Plot models comparation
def models_comparation(csv: pd.DataFrame,xlabel='Recall',ylabel='Models',title='Models comparation',del_last=True, selected_colors='Blues'):
    '''
    Objetivo:
    ---
    Comparar mediante un histograma las métricas obtenidas con los distintos modelos de clasificación.

    args.
    ---
    csv: dataframe
    xlabel: string
    ylabel: string
    title: string
    del_last: boolean; eliminar la última fila del dataframe
    selected_colors: string

    ret.
    ---
    Gráfica de comparación
    '''

    saved_metrics = csv.copy()

    cv_means = saved_metrics['Recall']
    lista = saved_metrics['model']

    if del_last:
        cv_means = cv_means[:-1]
        lista = lista[:-1]  

    cv_frame = pd.DataFrame(
        {
            "CrossValMeans":cv_means.astype(float),
            "Models": lista
        })

    cv_plot = sns.barplot(data = cv_frame,x="CrossValMeans",y="Models", palette=selected_colors)

    cv_plot.set_xlabel(xlabel,fontweight="bold")
    cv_plot.set_ylabel(ylabel,fontweight="bold")
    cv_plot = cv_plot.set_title(title,fontsize=16,fontweight='bold')  

def choose_params(model: str,clf: bool = True):
    '''
    Objetivo: 
    ---
    Elegir los parámetros a probar para un modelo concreto.

    *args.
    ----
    model: modelo del cual se quieren los parámetros.
    clf: bool; True: si se trata de un modelo de clasificación. 

    *ret.
    ----
    dict; con los parámetros a probar.

    NOTA: algunos parámetros pueden deprecarse, si actualizan la versión de la librería y/o el modelo

    '''
    if clf :

        clf_params = {

            'LogReg' : {

                'penalty' : ['l1','l2','elasticnet','none'],
                'class_weight' : ['none','balanced'],
                'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                'max_iter' : [50,75,100,150,200]
            },

            'KNNC' : {

                'n_neighbors' : [3,5,7,9,11,13,15],
                'weights' : ['uniform','distance'],
                'algorithm' : ['ball_tree','kd_tree','brute','auto'],
                'leaf_size' : [20,30,40],
                'p' : [1,2]

            },

            'DTC' : {
                
                'criterion' : ['log_loss','gini','entropy'],
                'splitter' : ['best','random'],
                'max_depth' : [7,9,11,13,None],
                'max_features': ['log2','sqrt'],
                'class_weight' : [None,'balanced']

            },

            'RFC' : {
                'n_estimators': np.linspace(10,150,10).astype(int),
                'criterion': ['gini','entropy'],
                'max_depth' : [7,9,11,13,None],
                'max_features': ['log2','sqrt',None],
                'class_weight' : [None,'balanced']
            },

            'BagC' : {
                'estimator__class_weight': ['balanced'],
                'n_estimators' : [10, 20, 30, 50, 100],
                'max_samples' : [0.05, 0.1, 0.2, 0.5]
            },
            'AdaBC' : {
                'base_estimator__class_weight': ['balanced'],
                'n_estimators' : [10, 20, 30, 50, 100]
            
            },

            'GBC' : [{
                'learning_rate' : [0.1, 0.05, 0.01, 0.001],
                'n_estimators' : [10, 20, 30, 50, 100],
                'max_depth' : [7,9,11,13,None],
                'criterion': ['friedman_mse','mse'],
                'loss': ['log_loss','exponential']
            },
            {
              'loss' : ["log_loss"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01, 0.001],
              'max_depth': [4, 8,16],
              'min_samples_leaf': [100,150,250],
              'max_features': [0.3, 0.1]
              }
            ],

            'SVC' : [
                {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'class_weight' : [None,'balanced']},
                {'C': [1, 10, 100, 1000],  'kernel': ['rbf'],'class_weight' : [None,'balanced']}
            ],

            'XGBC' : {
                'nthread':[4], 
                'objective':['binary:logistic'],
                'learning_rate': [0.05],
                'max_depth': [4,5,6,7],
                'min_child_weight': [1, 5, 10, 11],
                'subsample': [0.6,0.8,1.0],
                'colsample_bytree': [0.6,0.7,1.0],
                'n_estimators': [5,50,100], 
                'missing':[-999],
                'seed': [1337]
            },
        }

        return clf_params[model]

    else :

        reg_params = {

            'LinReg' : {},
            'KNNR' : {},
            'GNBR' : {},
            'BNBR' : {},
            'ENR' : {},
            'DTR' : {},
            'ETR' : {},
            'RFR' : {},
            'BagR' : {},
            'AdaBR' : {},
            'GBR' : {},
            'SVR' : {},
            'XGBR' : {}
        }

        return reg_params[model]

def choose_models(model: str, params = None, clf: bool = True):
    '''
    Objetivo: 
    ---
    Elegir el modelo o los modelos que correspondan.

    *args.
    ----
    model: str; modelo que se quiere seleccionar. 
        'all': selecciona todos los modelos. 

    *ret.
    ----
    El/los modelos seleccionados.

    '''
    
    if clf :
        if params == None:

            classification_models={

                'LogReg' : LogisticRegression(),
                'KNNC' : KNeighborsClassifier(),
                'DTC' : DecisionTreeClassifier(),
                'RFC' : RandomForestClassifier(),
                'BagC' : BaggingClassifier(), 
                'AdaBC' : AdaBoostClassifier(),
                'GBC' : GradientBoostingClassifier(),
                'SVC' : SVC(),
                'XGBC' : XGBClassifier(),
                'VC': VotingClassifier(estimators=[('RFC',RandomForestClassifier())]),
                'LDA': LinearDiscriminantAnalysis()
            }

        else:
            classification_models={

                'LogReg' : LogisticRegression(params),
                'KNNC' : KNeighborsClassifier(params),
                'DTC' : DecisionTreeClassifier(params), # type: ignore
                'RFC' : RandomForestClassifier(params),
                'BagC' : BaggingClassifier(params), 
                'AdaBC' : AdaBoostClassifier(params),
                'GBC' : GradientBoostingClassifier(params), # type: ignore
                'SVC' : SVC(params), # type: ignore
                'XGBC' : XGBClassifier(params),
                'VC': VotingClassifier(params),
                'LDA': LinearDiscriminantAnalysis(params)
            }


        if model == 'all' and params == None:
            return classification_models

        else:
            return classification_models[model]

    else : 

        if params == None:

            regression_models={

                'LinReg' : LinearRegression(),
                'KNNR' : KNeighborsRegressor(),
                'GNBR' : GaussianNB(),
                'BNBR' : BernoulliNB(),
                'ENR' : ElasticNet(),
                'DTR' : DecisionTreeRegressor(),
                'RFR' : RandomForestRegressor(),
                'BagR' : BaggingRegressor(), 
                'AdaBR' : AdaBoostRegressor(),
                'GBR' : GradientBoostingRegressor(),
                'SVR' : SVR(),
                'XGBR' : XGBRegressor()
                
            }

        else:

            regression_models={

                'LinReg' : LinearRegression(params),# type: ignore
                'KNNR' : KNeighborsRegressor(params),
                'GNBR' : GaussianNB(params), # type: ignore
                'BNBR' : BernoulliNB(params), # type: ignore
                'ENR' : ElasticNet(params),
                'DTR' : DecisionTreeRegressor(params), # type: ignore
                'RFR' : RandomForestRegressor(params),
                'BagR' : BaggingRegressor(params), 
                'AdaBR' : AdaBoostRegressor(params),
                'GBR' : GradientBoostingRegressor(params), # type: ignore
                'SVR' : SVR(params), # type: ignore
                'XGBR' : XGBRegressor(params)
                
            }

        if model == 'all'and params == None:
            return regression_models

        else:
            return regression_models[model]

def dataset_to_train_test(df: pd.DataFrame, test_size: float, random_state: int, path_train: str, path_test: str):
    
    """
    Objectivo:
    ---
    Divide un dataset en conjuntos de entrenamiento y prueba, y los guarda en archivos CSV.

    args.
    ---
    df: DataFrame; el DataFrame que se desea dividir y guardar.

    test_size: float; el tamaño del conjunto de prueba. Debe estar entre 0 y 1.

    random_state: int; semilla aleatoria para reproducibilidad de los resultados.

    ruta_train: str; ruta donde se guardará el archivo CSV del conjunto de entrenamiento.

    ruta_test: str; ruta donde se guardará el archivo CSV del conjunto de prueba.
    """

    # Dividir el dataset en conjuntos de entrenamiento y prueba
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    # Guardar los conjuntos de entrenamiento y prueba en archivos CSV
    train_df.to_csv(path_train, index=False)
    test_df.to_csv(path_test, index=False)

    print("Los conjuntos de entrenamiento y prueba se han guardado exitosamente.")

def baseline(X_train: pd.DataFrame,y_train: pd.Series,X_val: pd.DataFrame,y_val: pd.Series,
             base_model = None, clf: bool = True, file_name: str = 'metrics.csv', dir_file: str = 'model/model_metrics'):
    '''
    Objetivo:
    ---
    Evaluar el modelo base.

    args.
    ---
    X_train: pd.DataFrame; el conjunto de entrenamiento.
    y_train: pd.Series; etiquetas del conjunto de entrenamiento.
    X_val: pd.DataFrame; el conjunto de validación.
    y_val: pd.Series; etiquetas del conjunto de validación.
    base_model: sklearn.base.BaseEstimator; modelo base.
    clf: bool; indica si se quiere clasificación o regresión.
    file_name: str; nombre del archivo de salida.
    dir_file: str; ruta donde se guardará el archivo de salida.

    ret.
    ---
    y_pred: pd.Series; predicciones del conjunto de validación.
    metrics: dict; diccionario de métricas.
    estimator: sklearn.base.BaseEstimator; modelo base.
    '''
    if base_model == None:
        if clf:
            base_model = RandomForestClassifier()
        else:
            base_model = RandomForestRegressor()

    estimator=base_model.fit(X_train,y_train)
    y_pred=estimator.predict(X_val)
    metrics = eval_metrics(y_pred,y_val,clf) # type: ignore
    model_str = str(base_model)[0:str(base_model).find('(')]

    dict4save(metrics, file_name, dir_file, addcols=True, cols='model', vals=model_str,sep=';')
    return y_pred, metrics, estimator

def train_predict_best_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, model, params: dict, scoring: dict, 
                              random: int = 77, scaling: bool = False, balancing: bool = False, scaler = None):
    '''
    Objetivo: 
    ---
    Entrenar el modelo con los mejores parámetros introducidos y predecir con este.

    args.
    ---
    X_train: pd.DataFrame; conjunto de datos de entrenamiento.
    y_train: pd.Series; etiquetas de entrenamiento.
    X_test: pd.DataFrame; conjunto de datos de prueba.
    model: pd.BaseEstimator; modelo a entrenar.
    params: dict; conjunto de parámetros para modificar y probar mediante GridSearchCV.
    scoring: dict; métrica(s) a optimizar en el GridSearchCV.
    tsize: float; tamaño en tanto por uno del test. (Por defecto: 0.2)
    random: int; parámetro elegido para randomizar. (Por defecto: 77)
    scaling: bool; True para escalar y False para no escalar los datos. 
    scaler: None si se realiza el escalado generado/entrenado en la propia función y el escalador entrenado si se pretende usar uno concreto ya preentrenado.
    balancing: bool; True para balancear los datos y Falso si no se requiere.

    ret.
    ---
    estimator, X_test, y_test, X_train, y_train, y_pred

    '''
    
    # Escalado:
    if scaling:
        if scaler == None:
            scaler = StandardScaler().fit(X_train)
        X_train = list(scaler.transform(X_train))  # type: ignore
        X_test = scaler.transform(X_test)  # type: ignore
    
    
    # Balanceo
    if balancing:
        sm = SMOTEENN(random_state = random) 
        X_train, y_train = sm.fit_resample(X_train, y_train) # type: ignore

        

    # Entrenando al modelo: 
    estimator = GridSearchCV(model, params, scoring = scoring, refit = 'Recall', return_train_score = True)
    estimator.fit(X= X_train,y=y_train) # type: ignore
    y_pred = estimator.predict(X_test) # type: ignore

    return estimator,  y_pred

def models_generator(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, model = None, params = None, clf = True, scaling = True, scaler=None, scoring = {"AUC": "roc_auc", "Accuracy": make_scorer(accuracy_score)}, balancing = False, file_name = 'metrics.csv', dir_file = 'model/model_metrics', dir_model_file = 'model', random = 77):
    '''
    Objetivo: 
    ---
    Entrenar un modelo, evaluar sus métricas y guardar los resultados obtenidos en el archivo indicado.

    args.
    ----
    X_train: pd.DataFrame; conjunto de datos de entrenamiento.
    y_train: pd.Series; etiquetas de entrenamiento.
    X_test: pd.DataFrame; conjunto de datos de prueba.
    y_test: pd.Series; etiquetas de prueba.
    model: estimador que se va a utilizar. Predeterminadamente se utilizar RandomForest(). (opcional)
    params: parámetros que se prueban para obtener el mejor modelo mediante el GridSearchCV. (opcional)
    clf: True/False; si es un dataset de clasificación (True) si es de regresión (False). (opcional)
    tsize: float; tamaño del test [0.0,1.0]. (opcional)
    random: int; random state, semilla. (opcional)
    scaling: bool; True para escalar y False para no escalar los datos. 
    scaler: None si se realiza el escalado generado/entrenado en la propia función y el escalador entrenado si se pretende usar uno concreto ya preentrenado.
    scoring: dict; métrica(s) a optimizar en el GridSearchCV.
    balancing: bool; True para balancear los datos y Falso si no se requiere.


    ret.
    ----
        model_pack = {

            'trained_model' : estimator,
            'Xytest' : [X_test, y_test],
            'Xytrain' : [X_train, y_train],
            'ypred' : y_pred
        }

    '''

    # Modelo por defecto:
    if model == None:
        if clf:
            model = RandomForestClassifier()
        else:
            model = RandomForestRegressor()

    # Estimador entrenado y predicción: 
    estimator, y_pred = train_predict_best_model(X_train, y_train, X_test, model, params, scoring, # type: ignore
                                                  random = random, scaling = scaling, balancing = balancing, scaler=scaler)

    # Evaluación de métricas:
    metrics = eval_metrics(y_pred,y_test,clf) # type: ignore
    
    # Guardar modelo y métricas obtenidas:
    save_all(model, estimator, params, metrics, file_name = file_name, dir_file = dir_file, dir_model_file = dir_model_file)

    # Variable de salida: 
    model_pack = {

        'trained_model' : estimator,
        'Xytest' : [X_test, y_test],
        'Xytrain' : [X_train, y_train],
        'ypred' : y_pred,
        'metrics' : metrics
    }

    return model_pack

#              ###
# **Funciones para trabajar con archivos**
#              ###

def save_file(file: str, head: str, content: list, dir = 'data', sep = ';'):
    '''
    Objetivo: archivar datos en un .csv o .txt provenientes de una lista. 

    args.
    ----
    file: str; nombre del archivo.
    head: str; cabecera/columnas del archivo separadas por ';'.
    content: list; contenido a almacenar.

    ret.
    ----
    No devuelve nada, solo realiza el guardado. 
    '''
    # Comprobar que existe el fichero/ directorio
    # print(os.getcwd())
    ruta_dir=os.path.join(os.getcwd(), dir)
    os.makedirs(ruta_dir,exist_ok=True)
    ruta_file=os.path.join(ruta_dir,f'{file}')
    if os.path.exists(ruta_file):
        # Crear directorio y fichero si no existe
        with open(ruta_file, mode='a',newline='\n') as out:
            # Guardar la informacion
            # print(type(content))
            [out.write(cont+sep) if (content[-1]!=cont) else out.write(cont) for cont in content]
            out.write('\n')
    
    else:   
        with open(ruta_file, mode='w') as out:
            out.write(head+'\n')
            print(type(content))

            # Guardar la informacion
            [out.write(cont+sep) if (content[-1]!=cont) else out.write(cont) for cont in content]
            out.write('\n')

def dict4save(dict: dict, name_file: str, dirf: str, addcols: bool = False, cols: str = 'new cols', vals: str = 'values cols added', sep: str = ';'):
    '''
    Objetivo:
    ---
    Guarda los valores obtenidos como dict en el .csv o .txt.

    args.
    ---
    dict: dict; diccionario con los valores a guardar.
    name_file: str; nombre del archivo donde se van a guardar los datos.
    dirf: str; nombre del directorio o ruta relativa donde está el archivo.
    addcols: bool; True: si queremos añadir columnas extras a las del diccionario. 
                   False: si no.
    cols: str; nombre de la(s) columna(s). (opcional; si hay más de una utilizar el separador)
    vals: str; valores de la(s) nueva(s) columna(s). (opcional; si hay más de una utilizar el separador)
    sep: str; separador utilizado en el archivo .csv o .txt

    ret.
    ---
    print: Saved cuando termina.
    '''

    str_dict=''

    for str_k in list(dict.keys()):
        str_dict = str_dict+str_k+sep
    values = [str(val) for val in dict.values()]
    
    if addcols:
        values.insert(0,vals)
        save_file(name_file,cols+sep+str_dict[:-1],values,dir = dirf)
    else:
        save_file(name_file,str_dict[:-1],values,dir = dirf)
    
    print('Saved')

def save_model(model,dirname: str):
    '''
    Objetivo: 
    ---
    Guardar el modelo en la carpeta elegida.

    args.
    ---
    model: modelo a guardar.
    dirname: str; ruta relativa a la carpeta donde se pretende guardar el modelo.

    ret.
    ---
    Realiza un print indicando que el modelo introducido ha sido guardado.

    Devuelve la ruta relativa del modelo.
    '''
    model_str = str(model)
    model_str = model_str[0:model_str.find('(')]
    ruta_dir = os.path.join(os.getcwd(), dirname)
    
    os.makedirs(ruta_dir,exist_ok=True)
    ruta_file = os.path.join(ruta_dir,f'{model_str}.pkl')
    
    
    if os.path.exists(ruta_file):
        for i in range(1,99):
            ruta_file = os.path.join(ruta_dir,f'{model_str}_{i}.pkl')
             
            if os.path.exists(ruta_file):
                x='otro intento'
            else:
                pickle.dump(model, open(ruta_file,'wb'))
                the_path = os.path.join(dirname,f'{model_str}_{i}.pkl')
                break
    else:
        pickle.dump(model, open(ruta_file,'wb'))
        the_path = os.path.join(dirname,f'{model_str}.pkl')

    print(f'Model {model_str} saved')
    
    return the_path # type: ignore

def save_all(model, estimator, params, metrics, 
             file_name = 'metrics.csv', dir_file = 'model/model_metrics', dir_model_file = 'model'):
    '''
    Objetivo:
    ---
    Guardar todos los resultados de un modelo.

    args.
    ---
    model: BaseEstimator; modelo a entrenar.
    estimator: GridSearchCV; modelo entrenado.
    params: dict; conjunto de parámetros para modificar y probar mediante GridSearchCV.
    metrics: dict; métrica(s) a optimizar en el GridSearchCV.
    file_name: str; nombre del archivo a guardar.
    dir_file: str; ruta donde se guardará el archivo.
    dir_model_file: str; ruta donde se guardará el archivo del modelo.

    '''
    model_str = str(model)[0:str(model).find('(')]
    
    file2save = {'model':model_str,'params_tried': str(params),'best_params':str(estimator.best_params_)}
    file2save.update(metrics)
    
    # Guardar modelo:
    model_path = save_model(estimator.best_estimator_,dir_model_file)

    file2save.update({'model_path' : model_path})

    #Guardar archivo:
    dict4save(file2save, file_name, dir_file, addcols=False,sep=';')
    print(f'{file_name} saved')