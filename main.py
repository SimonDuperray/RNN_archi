from Merger import Merger
from RNN import RNN_model
from Analyzer import Analyzer

# ==================== 
#       MERGER
# ====================
merger = Merger()
merger.run()

# ==================== 
#        PP+RNN
# ====================
rnn_model = RNN_model()
rnn_model.set_percent(merger.get_percent())

# ==================== 
#  ITERATIVE TRANINGS
# ====================
# hyperparameters
epochs = [5, 10, 15, 20, 25, 30]

for epo in epochs:
   rnn_model.run(
      nb_epochs=epo,
      batch_size=300,
      validation_split=0.2
   )

# ==================== 
#       ANALYZIS
# ====================
analyzer = Analyzer()
analyzer.set_epochs_list(epochs)
analyzer.analyze_epochs()