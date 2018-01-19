# Cifar10
5Layered - CNN on the CIFAR10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html) using TF Records

--- Realizzata in Tensorflow v1.0 ---

A regime supera il 70% di accuracy su test e validation, migliorabili con dei tweak (sopratutto riguardanti la keep probability di dropout)

# eseguire i seguenti script nella cartella origine della repository

#scarica e converti in TFRecord il dataset
#questa parte dello script richiede python 2.7
```
python2 Import_Cifar10.py
```

#esegui training e evalutazione
```
python model_run.py
```

#eseguire nella cartella della repository tensorboard per vedere le accuracies col seguente comando (in dump ci sono delle versioni già eseguite in un file HTML)
```
tensorboard --logdir=./log --debug
```


#I dettagli di architettura e estrazione dei dati sono in `CifarCNN_Architecture.py` e `TFR_Cifar10_load.py`

#in caso di problemi con tensorboard model_run scrive ogni 100 batches le accuracies di training validation e test nei rispettivi files nella cartella dump


# Previous Run
`dump/tensorboard.html`
#contiene le accuracy di training test e validation
