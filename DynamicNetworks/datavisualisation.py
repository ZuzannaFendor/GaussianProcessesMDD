import numpy as np

from AnalysisUtils import data_processing_pcg as datapcg
from AnalysisUtils import plot_pcg as plotpcg
import matplotlib.pyplot as plt


data, col_names, scl_data = datapcg.import_ESMdata()


variable_names = data.columns.values.tolist()
print(variable_names)
dosage_data = data.iloc[:, [0,1,2]]
print(dosage_data.columns)
dosage_data = dosage_data.drop_duplicates("dayno")

x = dosage_data["dayno"]  #fake data
y = dosage_data["concentrat"]
phases = dosage_data.groupby("phase").first()
colors = ['dimgray','gray','darkgray','silver','lightgray']
colors = ['yellow','blue','green','red','purple']
fig, ax = plt.subplots()
ax.set_ylim([0,200])

ax.set_title("dosage throughout the phases")
ax.legend(loc='upper right')
ax.plot(x, y, 'k')
ax.grid()
ax.margins(0) # remove default margins (matplotlib verision 2+)
oldph = int(phases.iloc[0]["dayno"])
labels = []
for i, col in enumerate(colors):
    labels.append("phase" + str(i+1))
    if i +1 < 5:
        ax.axvspan(oldph, int(phases.iloc[i+1]["dayno"]), facecolor=col, alpha=0.5, label="phase"+str(i))
        oldph = int(phases.iloc[i+1]["dayno"])
    else:
        ax.axvspan(oldph, dosage_data.iloc[-1]["dayno"], facecolor=col, alpha=0.5, label="phase"+str(i))

handles, _ = ax.get_legend_handles_labels()

# Slice list to remove first handle
plt.legend(handles = handles, labels = labels)
plt.show()


