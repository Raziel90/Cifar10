# Cifar10
5Layered - CNN on the CIFAR10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html) using TF Records

# eseguire i seguenti script nella cartella origine della repository

#scarica e converti in TFRecord il dataset

python2 Import_Cifar10.py 

#esegui training e evalutazione
python model_run.py

#eseguire nella cartella tensorboard per vedere le accuracies (in dump ci sono delle versioni già eseguite in un file HTML)
tensorboard --logdir=$PWD --debug


#I dettagli di architettura e estrazione dei dati sono in CifarCNN_Architecture.py e TFR_Cifar10_load.py

#in caso di problemi con tensorboard model_run scrive ogni 100 batches le accuracies di training validation e test nei rispettivi files nella cartella dump
