import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dot, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

num_users = 10
num_posts = 15
embedding_size = 5

np.random.seed(42) 
user_ids = np.random.randint(1, num_users+1, size=100)
post_ids = np.random.randint(1, num_posts+1, size=100)
interactions = np.random.randint(0, 2, size=100)

interactions_df = pd.DataFrame({'user_id': user_ids, 'post_id': post_ids, 'interaction': interactions})

user_input = Input(shape=(1,), name='user_input')
post_input = Input(shape=(1,), name='post_input')

user_embedding = Embedding(num_users+1, embedding_size, input_length=1, name='user_embedding', embeddings_regularizer=l2(0.001))(user_input)
post_embedding = Embedding(num_posts+1, embedding_size, input_length=1, name='post_embedding', embeddings_regularizer=l2(0.001))(post_input)

merged_vectors = Dot(axes=2)([user_embedding, post_embedding])
flattened_vectors = Flatten()(merged_vectors)
dropout_layer = Dropout(0.5)(flattened_vectors)
output = Dense(1, activation='sigmoid')(dropout_layer)

model = Model(inputs=[user_input, post_input], outputs=output)
model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(interactions_df[['user_id', 'post_id']], interactions_df['interaction'], test_size=0.2, random_state=42)

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
model.fit([X_train['user_id'], X_train['post_id']], y_train, epochs=100, batch_size=10, validation_split=0.2, callbacks=[early_stopping])

y_pred = model.predict([X_test['user_id'], X_test['post_id']])
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error on Test Set:', mse)

model.save('groupbuy_recommendation_model_tf.h5') 

# new_model = tf.keras.models.load_model('groupbuy_recommendation_model_tf.h5')
