class CustomTransformation:

   def __init__(self):
      pass
   
   def transform(self, li):
      to_return = []
      for i in range(len(li)):
         to_return.append('author:'+"'"+li[i][17:-3]+"'")
      return to_return