import numpy as np
import matplotlib.pyplot as plt

from scipy import signal

u = {}
u['frequency'] = 0.01 # Hz
u['fill'] = -1
u['buffer'] = 10
u['chronic'] = lambda timepoints: signal.square(2*np.pi*u['frequency']*timepoints)
u['exposure'] = lambda timepoints: np.lib.pad(u['chronic'](t)[:int(1/u['frequency'])],
											  (u['buffer'],len(timepoints)-int(1/u['frequency']+u['buffer'])),
											  'constant',constant_values=(u['fill'],u['fill']))
u['cessation'] = 

t = np.linspace(0,1000,num=1001)

buff = 10
pulse = u['chronic'](t)[:int(1/u['frequency'])]
fill = min(pulse)
pulse = np.lib.pad(pulse,(buff,len(t)-(len(pulse)+buff)),'constant',constant_values=(fill,fill))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(u['exposure'](t),'k',linewidth=2)
ax.set_ylim((-1.5,1.5))
plt.show()
