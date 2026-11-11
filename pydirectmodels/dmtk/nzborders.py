import json
import matplotlib.pyplot as plt

def plot(file=None, plinewidth= 0.1, pcolor = 'b', psym = '-', ax=None):
    if file is None:
       f = open("src/nzborders/nzborders.json")
    else:
       f = open(file) 
    if ax is None:
       fig, ax = plt.subplots()
         
    border = json.load(f)
    f.close()
    # plt.figure(figsize=(7, 7))
    # plt.gca().set_aspect('equal')
    
    ax.plot(border['lon'], border['lat'], psym, linewidth= plinewidth, color=pcolor)
    ax.set_xlabel('Longitude (\u00B0E)')
    ax.set_ylabel('Latitude (\u00B0N)')
    
