from utils1 import *

def model_train():

    # Data PreProccessing
    nltk.download('punkt')
    from nltk.stem.lancaster import LancasterStemmer

    # loading Dataset
    intents = load_data('Dataset\intents.json')

    # Organinzing Dataset
    words,classes,documents = organize_Data(intents)

    # Generating training Data
    train_x,train_y = generate_dataset(words,classes,documents)

    # SaVing training Data as pkl files for later usage
    train_and_save_model('./pickle_files',words,classes,train_x,train_y)

    num_folds = 2
    kf = KFold(n_splits=num_folds, shuffle=True)

    for train_index, test_index in kf.split(train_x):
        train_x_fold, test_x_fold = np.array(train_x)[train_index], np.array(train_x)[test_index]
        train_y_fold, test_y_fold = np.array(train_y)[train_index], np.array(train_y)[test_index]

        model = build_model(train_x, train_y)
        model.fit(train_x_fold, train_y_fold, epochs=200, batch_size=8, verbose=1)

    return model   

    # Save the model
    model.save('Medical-chatbot.h5')

  

    model_train()