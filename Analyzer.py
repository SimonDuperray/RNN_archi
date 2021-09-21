import matplotlib.pyplot as plt, os, json
from os import listdir
from os.path import isfile, join

class Analyzer:

   def __init__(self):
      self.epochs_list = None

   def set_epochs_list(self, li):
      self.epochs_list = li

   def analyze_epochs(self):
      list_of_dirs = listdir("./histories")
      results = []
      for dirr in list_of_dirs:
         dir_path = "/"+dirr
         files = listdir(join('./histories', dirr))
         if files[0][-4:]=="json":
            with open("./histories/"+dirr+"/"+files[0], 'r') as file:
               data = json.load(file)
               results.append(
                  {
                     "epochs": [i+1 for i in range(data['epochs'])],
                     "accuracy": data['accuracy'],
                     "val_accuracy": data['val_accuracy'],
                     "loss": data['loss'],
                     "val_loss": data['val_loss']
                  }
               )    

      # fill blanks with None
      epochs_list = []
      for item in results:
         epochs_list.append(len(item['epochs']))
      max_epochs = max(epochs_list)
      epochss = [i+1 for i in range(max_epochs)]

      # accuracy
      for item in results:
         plt.plot(epochss, item['accuracy']+[None for i in range(max_epochs-len(item['accuracy']))])
      plt.legend([str(i)+" epochs" for i in self.epochs_list])
      plt.title('Evolution of accuracy across epochs')
      plt.savefig("./analysis/compar_epochs/acc.png")
      plt.show()

      # val accuracy
      for item in results:
         plt.plot(epochss, item['val_accuracy']+[None for i in range(max_epochs-len(item['val_accuracy']))])
      plt.legend([str(i)+" epochs" for i in self.epochs_list])
      plt.title('Evolution of val_accuracy across epochs')
      plt.savefig("./analysis/compar_epochs/val_acc.png")
      plt.show()

      # loss
      for item in results:
         plt.plot(epochss, item['loss']+[None for i in range(max_epochs-len(item['loss']))])
      plt.legend([str(i)+" epochs" for i in self.epochs_list])
      plt.title('Evolution of loss across epochs')
      plt.savefig("./analysis/compar_epochs/loss.png")
      plt.show()

      # val loss
      for item in results:
         plt.plot(epochss, item['val_loss']+[None for i in range(max_epochs-len(item['val_loss']))])
      plt.legend([str(i)+" epochs" for i in self.epochs_list])
      plt.title('Evolution of val_loss across epochs')
      plt.savefig("./analysis/compar_epochs/val_loss.png")
      plt.show()
               
      # accuracy / val_accuracy
      for i in range(len(results)): 
         plt.plot(results[i]['epochs'], results[i]['accuracy'])
         plt.plot(results[i]['epochs'], results[i]['val_accuracy'])
         plt.xlabel('epochs')
         title = "acc/val_acc for "+str(self.epochs_list[i])+" epochs"
         plt.title(title)
         plt.legend(['acc', 'val_acc'])
         filename = "comp_acc_valacc_"+str(self.epochs_list[i])+".png"
         path = "./analysis/compar_epochs/"+filename
         plt.savefig(path)
         plt.show()