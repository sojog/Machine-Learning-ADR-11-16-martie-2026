import tensorflow as tf

augumentation_layer = tf.keras.models.Sequential([
    ## Date Augumentate
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
    # Modificare 1: Am schimbat RandomFlip. 
    # CIFAR-10 conține imagini reale (mașini, animale). Nu vrem să le întoarcem pe verticală 
    # (cu susul în jos) pentru că modelul s-ar confunda.
    tf.keras.layers.RandomFlip("horizontal"),
])

cnn_layer = tf.keras.models.Sequential([
    # Modificare 2: Arhitectură mai adâncă de tip VGG, adăugare de Padding, BatchNormalization și Dropout
    # Bloc 1 de convoluție (32 filtre)
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(), # Normalizează activările și accelerează convergența
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25), # Regularizare spațială (aruncăm informație pentru a forța generalizarea)

    # Bloc 2 de convoluție (64 filtre)
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.35),

    # Bloc 3 de convoluție (128 filtre)
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.45),
])

fc_layer = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),

    # Modificare 3: Am redus unitățile din Dense pentru a preveni overfitting-ul major de la 512+256+128.
    # Un singur strat consistent (ex: 256) împreună cu Dropout mare este mult mai eficient.
    tf.keras.layers.Dense(units=256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5), # 50% șanse ca neuronul curent să fie dezactivat pentru generalizare
    
    ## 10 categorii, functia de activa este softmax
    tf.keras.layers.Dense(units=10, activation='softmax'),
])

arhitectura_marita = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),

    ## LAYER DE AUGUMENTARE
    augumentation_layer,

    ## LAYER DE Rescalare  - > imaginea de la 0..255 devine 0..1
    tf.keras.layers.Rescaling(scale=1./255),

    ## CNN
    cnn_layer,

    ### FULLY CONNECTED LAYER
    fc_layer
])

# Am explicitat Learning Rate-ul la compilare
arhitectura_marita.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)
