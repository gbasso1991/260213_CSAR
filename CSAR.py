#%% CSAR NE & NF 260203
#%% ===================== IMPORTS =====================

import os
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from uncertainties import ufloat, unumpy
from scipy.optimize import curve_fit

#%% Lector Templog
def lector_templog(path):
    '''
    Busca archivo *templog.csv en directorio especificado.
    muestras = False plotea solo T(dt). 
    muestras = True plotea T(dt) con las muestras superpuestas
    Retorna arrys timestamp,temperatura 
    '''
    data = pd.read_csv(path,sep=';',header=5,
                            names=('Timestamp','T_CH1','T_CH2'),usecols=(0,1,2),
                            decimal=',',engine='python') 
    temp_CH1  = pd.Series(data['T_CH1']).to_numpy(dtype=float)
    temp_CH2  = pd.Series(data['T_CH2']).to_numpy(dtype=float)
    timestamp = np.array([datetime.strptime(date,'%Y/%m/%d %H:%M:%S') for date in data['Timestamp']]) 

    return timestamp,temp_CH1, temp_CH2
#%% Levanto data
path_agua='260213_111319_agua_300_150.csv'
t_agua,T_agua,_=lector_templog(path_agua)
t_agua_0 = np.array([(t-t_agua[0]).total_seconds() for t in t_agua])
t_agua_on=datetime(2026, 2, 13, 11, 14, 00)
t_agua_off = datetime(2026, 2, 13, 11, 19, 00)
t_agua_medida=t_agua[(t_agua>=t_agua_on) & (t_agua<=t_agua_off)]
tiempo_medida_agua=(t_agua_off - t_agua_on).total_seconds()

T_agua_medida = T_agua[(t_agua>=t_agua_on) & (t_agua<=t_agua_off)]

#para T_eq promedio los valores de la medida pesandolos por tiempo de cada medida
T_eq_agua = np.mean(T_agua_medida)
print(f'Teq agua = {T_eq_agua:.2f} °C')

plt.figure(figsize=(12,5),constrained_layout=True)
plt.plot(t_agua,T_agua,'o-')
plt.plot(t_agua_medida,T_agua_medida,'.-',label='Medida')
plt.axvline(x=t_agua_on, color='g', ls='-',label=f't inicio = {t_agua_on}')
plt.axvline(x=t_agua_off, color='r', ls='-',label=f't corte = {t_agua_off}')
plt.axhline(y=T_eq_agua, color='k', ls='--',label=f'Teq = {T_eq_agua:.2f} °C')
plt.legend(title=f'Tiempo de medida = {tiempo_medida_agua} s')
plt.ylabel('T (°C)')
plt.ylim(23.5,24.5)
plt.grid()      
plt.title('Agua - 300 kHz - 57 kA/m')
plt.show()

#%% NE
paths_NE_300_150=glob('**/*NE_300_150.csv', recursive=True)
paths_NE_300_125=glob('**/*NE_300_125.csv', recursive=True)
paths_NE_300_100=glob('**/*NE_300_100.csv', recursive=True)
paths_NE_300_075=glob('**/*NE_300_075.csv', recursive=True)
paths_NE_300_050=glob('**/*NE_300_050.csv', recursive=True)
paths_NE = [paths_NE_300_150,paths_NE_300_125,paths_NE_300_100,paths_NE_300_075,paths_NE_300_050]
for p in paths_NE:
    p.sort()

paths_NE=np.array(paths_NE).flatten()

t_NE_300_150_1,T_NE_300_150_1,_=lector_templog(paths_NE_300_150[0])
t_NE_300_150_2,T_NE_300_150_2,_=lector_templog(paths_NE_300_150[1])

t_NE_300_125_1,T_NE_300_125_1,_=lector_templog(paths_NE_300_125[0])
t_NE_300_125_2,T_NE_300_125_2,_=lector_templog(paths_NE_300_125[1])

t_NE_300_100_1,T_NE_300_100_1,_=lector_templog(paths_NE_300_100[0])
t_NE_300_100_2,T_NE_300_100_2,_=lector_templog(paths_NE_300_100[1])

t_NE_300_075_1,T_NE_300_075_1,_=lector_templog(paths_NE_300_075[0])
t_NE_300_075_2,T_NE_300_075_2,_=lector_templog(paths_NE_300_075[1])

t_NE_300_050_1,T_NE_300_050_1,_=lector_templog(paths_NE_300_050[0])
t_NE_300_050_2,T_NE_300_050_2,_=lector_templog(paths_NE_300_050[1])

tiempos=[t_NE_300_150_1,t_NE_300_150_2,
         t_NE_300_125_1,t_NE_300_125_2,
         t_NE_300_100_1,t_NE_300_100_2,
         t_NE_300_075_1,t_NE_300_075_2,
         t_NE_300_050_1,t_NE_300_050_2]
temperaturas=[T_NE_300_150_1,T_NE_300_150_2,
             T_NE_300_125_1,T_NE_300_125_2,
             T_NE_300_100_1,T_NE_300_100_2,
             T_NE_300_075_1,T_NE_300_075_2,
             T_NE_300_050_1,T_NE_300_050_2]

t_on = [datetime(2026, 2, 13, 11, 31, 30), datetime(2026, 2, 13, 11, 40, 50),
        datetime(2026, 2, 13, 11, 47, 50), datetime(2026, 2, 13, 11, 57, 30),
        datetime(2026, 2, 13, 12, 7, 30), datetime(2026, 2, 13, 12, 17, 30),
        datetime(2026, 2, 13, 12, 29, 00), datetime(2026, 2, 13, 12, 42, 20),
        datetime(2026, 2, 13, 12, 54, 30), datetime(2026, 2, 13, 14, 32, 20)]

t_off = [datetime(2026, 2, 13, 11, 34, 20), datetime(2026, 2, 13, 11, 43, 50),
         datetime(2026, 2, 13, 11, 51, 00), datetime(2026, 2, 13, 12, 00, 40),
         datetime(2026, 2, 13, 12, 11, 30), datetime(2026, 2, 13, 12, 23, 00),
         datetime(2026, 2, 13, 12, 36, 00), datetime(2026, 2, 13, 12, 49, 20),
         datetime(2026, 2, 13, 13, 7, 30), datetime(2026, 2, 13, 14, 51, 10)]
delta_t = [(off-on).total_seconds() for on, off in zip(t_on, t_off)]

titulos=[57,57,47,47,38,38,28,28,19,19]
for i in range(0,len(titulos),2):  
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(11,9),constrained_layout=True)
    ax.plot(tiempos[i],temperaturas[i],'.-',label=f'{titulos[i]} kA/m - 300 kHZ')
    ax.axvline(x=t_on[i], color='g', ls='-',label=f't inicio = {t_on[i]}')
    ax.axvline(x=t_off[i], color='r', ls='-',label=f't corte = {t_off[i]}')
    ax.set_title(f'H$_0$ = {titulos[i]} kA/m - 300 kHZ',loc='left')
    ax2.plot(tiempos[i+1],temperaturas[i+1],'.-',label=f'{titulos[i]} kA/m - 300 kHZ')
    ax2.axvline(x=t_on[i+1], color='g', ls='-',label=f't inicio = {t_on[i+1]}')
    ax2.axvline(x=t_off[i+1], color='r', ls='-',label=f't corte = {t_off[i+1]}')
    
    ax.legend(title=f'Tiempo de medida = {delta_t[i]} s')
    ax2.legend(title=f'Tiempo de medida = {delta_t[i+1]} s')
    for a in (ax,ax2):
        a.grid()
        a.set_ylim(20,43)
        
    ax.set_title(paths_NE[i],loc='left')
    ax2.set_title(paths_NE[i+1],loc='left')
    ax2.set_xlabel('t (s)')  
    
    plt.suptitle(f'H$_0$ = {titulos[i]} kA/m - 300 kHZ')   
    plt.savefig(f'NE_T_vs_t_{titulos[i]}kAm_300kHz.png', dpi=300)
    plt.show()    

t_NE_300_150_1 = np.array([(t-t_NE_300_150_1[0]).total_seconds() for t in t_NE_300_150_1])
t_NE_300_150_2 = np.array([(t-t_NE_300_150_2[0]).total_seconds() for t in t_NE_300_150_2])
t_NE_300_125_1 = np.array([(t-t_NE_300_125_1[0]).total_seconds() for t in t_NE_300_125_1])
t_NE_300_125_2 = np.array([(t-t_NE_300_125_2[0]).total_seconds() for t in t_NE_300_125_2])
t_NE_300_100_1 = np.array([(t-t_NE_300_100_1[0]).total_seconds() for t in t_NE_300_100_1])
t_NE_300_100_2 = np.array([(t-t_NE_300_100_2[0]).total_seconds() for t in t_NE_300_100_2])
t_NE_300_075_1 = np.array([(t-t_NE_300_075_1[0]).total_seconds() for t in t_NE_300_075_1])  
t_NE_300_075_2 = np.array([(t-t_NE_300_075_2[0]).total_seconds() for t in t_NE_300_075_2])
t_NE_300_050_1 = np.array([(t-t_NE_300_050_1[0]).total_seconds() for t in t_NE_300_050_1])
t_NE_300_050_2 = np.array([(t-t_NE_300_050_2[0]).total_seconds() for t in t_NE_300_050_2])

fig,(ax,ax1,ax2,ax3,ax4)=plt.subplots(5,1,figsize=(10,10), constrained_layout=True,sharex=True,sharey=True)

ax.set_title('57 kA/m',loc='left')
ax1.set_title('47 kA/m',loc='left')
ax2.set_title('38 kA/m',loc='left')
ax3.set_title('28 kA/m',loc='left')
ax4.set_title('19 kA/m',loc='left')

ax.plot(t_NE_300_150_1,T_NE_300_150_1,'.-',label=paths_NE[0])
ax.plot(t_NE_300_150_2,T_NE_300_150_2,'.-',label=paths_NE[1])
ax1.plot(t_NE_300_125_1,T_NE_300_125_1,'.-',label=paths_NE[2])
ax1.plot(t_NE_300_125_2,T_NE_300_125_2,'.-',label=paths_NE[3])
ax2.plot(t_NE_300_100_1,T_NE_300_100_1,'.-',label=paths_NE[4])
ax2.plot(t_NE_300_100_2,T_NE_300_100_2,'.-',label=paths_NE[5])
ax3.plot(t_NE_300_075_1,T_NE_300_075_1,'.-',label=paths_NE[6])
ax3.plot(t_NE_300_075_2,T_NE_300_075_2,'.-',label=paths_NE[7])
ax4.plot(t_NE_300_050_1,T_NE_300_050_1,'.-',label=paths_NE[8])
ax4.plot(t_NE_300_050_2,T_NE_300_050_2,'.-',label=paths_NE[9])
for a in (ax,ax1,ax2,ax3,ax4):
    a.grid()
    a.legend()
    a.set_xlim(0,)
    a.set_ylim(20,43)
ax4.set_xlabel('t (s)')   
plt.suptitle('NE@citrico - coprecpitacion',fontsize=16)
plt.savefig('NE_T_vs_t_all.png', dpi=300)

plt.show()
#%% ===================== NF   =====================

paths_NF_300_150=glob('**/*NF_300_150.csv', recursive=True)
paths_NF_300_125=glob('**/*NF_300_125.csv', recursive=True)
paths_NF_300_100=glob('**/*NF_300_100.csv', recursive=True)
paths_NF_300_075=glob('**/*NF_300_075.csv', recursive=True)
paths_NF_300_050=glob('**/*NF_300_050.csv', recursive=True)

paths_NF = [paths_NF_300_150,paths_NF_300_125,paths_NF_300_100,paths_NF_300_075,paths_NF_300_050]
for p in paths_NF:
    p.sort()

paths_NF=np.array(paths_NF).flatten()

# Cargo datos

t_NF_300_150_1,T_NF_300_150_1,_=lector_templog(paths_NF_300_150[0])
t_NF_300_150_2,T_NF_300_150_2,_=lector_templog(paths_NF_300_150[1])

t_NF_300_125_1,T_NF_300_125_1,_=lector_templog(paths_NF_300_125[0])
t_NF_300_125_2,T_NF_300_125_2,_=lector_templog(paths_NF_300_125[1])

t_NF_300_100_1,T_NF_300_100_1,_=lector_templog(paths_NF_300_100[0])
t_NF_300_100_2,T_NF_300_100_2,_=lector_templog(paths_NF_300_100[1])

t_NF_300_075_1,T_NF_300_075_1,_=lector_templog(paths_NF_300_075[0])
t_NF_300_075_2,T_NF_300_075_2,_=lector_templog(paths_NF_300_075[1])

t_NF_300_050_1,T_NF_300_050_1,_=lector_templog(paths_NF_300_050[0])
t_NF_300_050_2,T_NF_300_050_2,_=lector_templog(paths_NF_300_050[1])

tiempos_NF=[t_NF_300_150_1,t_NF_300_150_2,
         t_NF_300_125_1,t_NF_300_125_2,
         t_NF_300_100_1,t_NF_300_100_2,
         t_NF_300_075_1,t_NF_300_075_2,
         t_NF_300_050_1,t_NF_300_050_2]

temperaturas_NF=[T_NF_300_150_1,T_NF_300_150_2,
             T_NF_300_125_1,T_NF_300_125_2,
             T_NF_300_100_1,T_NF_300_100_2,
             T_NF_300_075_1,T_NF_300_075_2,
             T_NF_300_050_1,T_NF_300_050_2]

# Horarios on/off
t_on_NF = [datetime(2026,2,13,15,2,22),datetime(2026,2,13,15,11,30),
datetime(2026,2,13,15,20,10),datetime(2026,2,13,15,29,30),
datetime(2026,2,13,15,36,50),datetime(2026,2,13,15,43,00),
datetime(2026,2,13,15,52,23),datetime(2026,2,13,16,0,30),
datetime(2026,2,13,16,8,0),datetime(2026,2,13,16,20,0)]

t_off_NF = [datetime(2026,2,13,15,3,22),datetime(2026,2,13,15,13,50),
datetime(2026,2,13,15,21,10),datetime(2026,2,13,15,30,30),
datetime(2026,2,13,15,38,10),datetime(2026,2,13,15,44,15),
datetime(2026,2,13,15,54,30),datetime(2026,2,13,16,2,30),
datetime(2026,2,13,16,16,0),datetime(2026,2,13,16,29,0)]
#%%
delta_t_NF = [(off-on).total_seconds() for on,off in zip(t_on_NF,t_off_NF)]
titulos_NF = [57,57,47,47,38,38,28,28,19,19]
import matplotlib.dates as mdates

for i in range(0,len(titulos_NF),2):  
    fig,(ax,ax2)=plt.subplots(2,1,figsize=(11,9),constrained_layout=True)
    ax.plot(tiempos_NF[i],temperaturas_NF[i],'.-',label=f'{titulos_NF[i]} kA/m - 300 kHZ')
    ax.axvline(x=t_on_NF[i], color='g', ls='-',label=f't inicio = {t_on_NF[i]}')
    ax.axvline(x=t_off_NF[i], color='r', ls='-',label=f't corte = {t_off_NF[i]}')
    ax2.plot(tiempos_NF[i+1],temperaturas_NF[i+1],'.-',label=f'{titulos_NF[i+1]} kA/m - 300 kHZ')
    ax2.axvline(x=t_on_NF[i+1], color='g', ls='-',label=f't inicio = {t_on_NF[i+1]}')
    ax2.axvline(x=t_off_NF[i+1], color='r', ls='-',label=f't corte = {t_off_NF[i+1]}')
    
    ax.legend(title=f'Tiempo de medida = {delta_t_NF[i]} s')
    ax2.legend(title=f'Tiempo de medida = {delta_t_NF[i+1]} s')
    for a in (ax,ax2):
        a.grid()
        a.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
    ax.set_title(paths_NF[i],loc='left')
    ax2.set_title(paths_NF[i+1],loc='left')
    
    plt.suptitle(f'H$_0$ = {titulos_NF[i]} kA/m - 300 kHZ')
    plt.savefig(f'NF_T_vs_t_{titulos_NF[i]}kAm_300kHz.png', dpi=300)
    plt.show()    
    

t_NF_300_150_1 = np.array([(t-t_NF_300_150_1[0]).total_seconds() for t in t_NF_300_150_1])
t_NF_300_150_2 = np.array([(t-t_NF_300_150_2[0]).total_seconds() for t in t_NF_300_150_2])
t_NF_300_125_1 = np.array([(t-t_NF_300_125_1[0]).total_seconds() for t in t_NF_300_125_1])
t_NF_300_125_2 = np.array([(t-t_NF_300_125_2[0]).total_seconds() for t in t_NF_300_125_2])
t_NF_300_100_1 = np.array([(t-t_NF_300_100_1[0]).total_seconds() for t in t_NF_300_100_1])
t_NF_300_100_2 = np.array([(t-t_NF_300_100_2[0]).total_seconds() for t in t_NF_300_100_2])
t_NF_300_075_1 = np.array([(t-t_NF_300_075_1[0]).total_seconds() for t in t_NF_300_075_1])
t_NF_300_075_2 = np.array([(t-t_NF_300_075_2[0]).total_seconds() for t in t_NF_300_075_2])
t_NF_300_050_1 = np.array([(t-t_NF_300_050_1[0]).total_seconds() for t in t_NF_300_050_1])
t_NF_300_050_2 = np.array([(t-t_NF_300_050_2[0]).total_seconds() for t in t_NF_300_050_2])
#%%
fig,(ax,ax1,ax2,ax3,ax4)=plt.subplots(5,1,figsize=(10,10), constrained_layout=True,
                                      sharex=True,sharey=True)

ax.set_title('57 kA/m',loc='left')
ax1.set_title('47 kA/m',loc='left')
ax2.set_title('38 kA/m',loc='left')
ax3.set_title('28 kA/m',loc='left')
ax4.set_title('19 kA/m',loc='left')

ax.plot(t_NF_300_150_1,T_NF_300_150_1,'.-',label=paths_NF[0])
ax.plot(t_NF_300_150_2,T_NF_300_150_2,'.-',label=paths_NF[1])
ax1.plot(t_NF_300_125_1,T_NF_300_125_1,'.-',label=paths_NF[2])
ax1.plot(t_NF_300_125_2,T_NF_300_125_2,'.-',label=paths_NF[3])
ax2.plot(t_NF_300_100_1,T_NF_300_100_1,'.-',label=paths_NF[4])
ax2.plot(t_NF_300_100_2,T_NF_300_100_2,'.-',label=paths_NF[5])
ax3.plot(t_NF_300_075_1,T_NF_300_075_1,'.-',label=paths_NF[6])
ax3.plot(t_NF_300_075_2,T_NF_300_075_2,'.-',label=paths_NF[7])
ax4.plot(t_NF_300_050_1,T_NF_300_050_1,'.-',label=paths_NF[8])
ax4.plot(t_NF_300_050_2,T_NF_300_050_2,'.-',label=paths_NF[9])
for a in (ax,ax1,ax2,ax3,ax4):
    a.grid()
    a.legend()
    a.set_xlim(0,)
    #.set_ylim(20,43)
ax4.set_xlabel('t (s)')    

plt.suptitle('NF@citrico - coprecpitacion',fontsize=16)
plt.savefig('NF_T_vs_t_all.png', dpi=300)

plt.show()
#%% Comparativa NE vs NF


fig,(ax,ax1,ax2,ax3,ax4)=plt.subplots(5,1,figsize=(10,10), constrained_layout=True,
                                      sharex=True,sharey=True)

ax.set_title('57 kA/m',loc='left')
ax1.set_title('47 kA/m',loc='left')
ax2.set_title('38 kA/m',loc='left')
ax3.set_title('28 kA/m',loc='left')
ax4.set_title('19 kA/m',loc='left')

ax.plot(t_NF_300_150_1,T_NF_300_150_1,'.-',label=paths_NF[0])
ax.plot(t_NF_300_150_2,T_NF_300_150_2,'.-',label=paths_NF[1])
ax.plot(t_NE_300_150_1,T_NE_300_150_1,'.-',label=paths_NE[0])
ax.plot(t_NE_300_150_2,T_NE_300_150_2,'.-',label=paths_NE[1])

ax1.plot(t_NF_300_125_1,T_NF_300_125_1,'.-',label=paths_NF[2])
ax1.plot(t_NF_300_125_2,T_NF_300_125_2,'.-',label=paths_NF[3])

ax1.plot(t_NE_300_125_1,T_NE_300_125_1,'.-',label=paths_NE[2])
ax1.plot(t_NE_300_125_2,T_NE_300_125_2,'.-',label=paths_NE[3])

ax2.plot(t_NF_300_100_1,T_NF_300_100_1,'.-',label=paths_NF[4])
ax2.plot(t_NF_300_100_2,T_NF_300_100_2,'.-',label=paths_NF[5])

ax2.plot(t_NE_300_100_1,T_NE_300_100_1,'.-',label=paths_NE[4])
ax2.plot(t_NE_300_100_2,T_NE_300_100_2,'.-',label=paths_NE[5])

ax3.plot(t_NF_300_075_1,T_NF_300_075_1,'.-',label=paths_NF[6])
ax3.plot(t_NF_300_075_2,T_NF_300_075_2,'.-',label=paths_NF[7])

ax3.plot(t_NE_300_075_1,T_NE_300_075_1,'.-',label=paths_NE[6])
ax3.plot(t_NE_300_075_2,T_NE_300_075_2,'.-',label=paths_NE[7])

ax4.plot(t_NF_300_050_1,T_NF_300_050_1,'.-',label=paths_NF[8])
ax4.plot(t_NF_300_050_2,T_NF_300_050_2,'.-',label=paths_NF[9])

ax4.plot(t_NE_300_050_1,T_NE_300_050_1,'.-',label=paths_NE[8])
ax4.plot(t_NE_300_050_2,T_NE_300_050_2,'.-',label=paths_NE[9])
for a in (ax,ax1,ax2,ax3,ax4):
    a.grid()
    a.legend()
    a.set_xlim(0,)
    #.set_ylim(20,43)
ax4.set_xlabel('t (s)')    
plt.xlim(0,1200)
plt.suptitle('NF@citrico - coprecpitacion',fontsize=16)
plt.savefig('NE&NF_T_vs_t_all.png', dpi=300)



#%%%  Calculo de CSAR vs t 

# dT/dt vs t
# --- Datos ---

t_1 = t_NF_300_150_1
T_1 = T_NF_300_150_1

t_2 = t_NF_300_125_1
T_2 = T_NF_300_125_1

t_3 = t_NF_300_100_1
T_3 = T_NF_300_100_1

t_4 = t_NF_300_075_1
T_4 = T_NF_300_075_1


#recorto a maximo valor
t_1 = t_1[:np.argmax(T_1)+1]
T_1 = T_1[:np.argmax(T_1)+1]
t_2 = t_2[:np.argmax(T_2)+1]
T_2 = T_2[:np.argmax(T_2)+1]
t_3 = t_3[:np.argmax(T_3)+1]
T_3 = T_3[:np.argmax(T_3)+1]
t_4 = t_4[:np.argmax(T_4)+1]
T_4 = T_4[:np.argmax(T_4)+1]

dT1 = np.gradient(T_1, t_1)
dT2 = np.gradient(T_2, t_2)
dT3 = np.gradient(T_3, t_3)
dT4 = np.gradient(T_4, t_4)

concentracion_NF=15.0 #g/L
CSAR_1 = dT1*4.186e3/concentracion_NF
CSAR_2 = dT2*4.186e3/concentracion_NF
CSAR_3 = dT3*4.186e3/concentracion_NF
CSAR_4 = dT4*4.186e3/concentracion_NF

fig,(ax,ax2,ax3)=plt.subplots(3,1,figsize=(10,12),constrained_layout=True,sharex=True) 
ax.plot(t_1,T_1,'.-',label=paths_NF[0])
ax.plot(t_2,T_2,'.-',label=paths_NF[2])
ax.plot(t_3,T_3,'.-',label=paths_NF[4])
ax.plot(t_4,T_4,'.-',label=paths_NF[6])

ax2.plot(t_1,dT1,'.-',label=paths_NF[0])
ax2.plot(t_2,dT2,'.-',label=paths_NF[2])
ax2.plot(t_3,dT3,'.-',label=paths_NF[4])
ax2.plot(t_4,dT4,'.-',label=paths_NF[6])

ax3.plot(t_1,CSAR_1,'.-',label=paths_NF[0])
ax3.plot(t_2,CSAR_2,'.-',label=paths_NF[2])
ax3.plot(t_3,CSAR_3,'.-',label=paths_NF[4])
ax3.plot(t_4,CSAR_4,'.-',label=paths_NF[6])
for a in [ax,ax2,ax3]:
    a.grid()
    a.legend()
    a.set_xlim(0,)
    #.set_ylim(20,43)
ax.set_ylabel('T (°C)')
ax2.set_ylabel('dT/dt (°C/s)')
ax3.set_ylabel('CSAR (W/g)')
ax3.set_xlabel('t (s)')
plt.suptitle('CSAR - NF@citrico - 15.0 g/L',fontsize=16)
plt.savefig('CSAR_NF.png', dpi=300)
plt.show()

#%% CSAR NE
t_1 = t_NE_300_150_1
T_1 = T_NE_300_150_1
t_2 = t_NE_300_125_1
T_2 = T_NE_300_125_1
t_3 = t_NE_300_100_1
T_3 = T_NE_300_100_1
t_4 = t_NE_300_075_1
T_4 = T_NE_300_075_1

#%
#recorto a maximo valor
t_1 = t_1[:np.argmax(T_1)+1]
T_1 = T_1[:np.argmax(T_1)+1]
t_2 = t_2[:np.argmax(T_2)+1]
T_2 = T_2[:np.argmax(T_2)+1]
t_3 = t_3[:np.argmax(T_3)+1]
T_3 = T_3[:np.argmax(T_3)+1]
t_4 = t_4[:np.argmax(T_4)+1]
T_4 = T_4[:np.argmax(T_4)+1]

dT1 = np.gradient(T_1, t_1)
dT2 = np.gradient(T_2, t_2)
dT3 = np.gradient(T_3, t_3)
dT4 = np.gradient(T_4, t_4)

concentracion_NF=15.0 #g/L
CSAR_1 = dT1*4.186e3/concentracion_NF
CSAR_2 = dT2*4.186e3/concentracion_NF
CSAR_3 = dT3*4.186e3/concentracion_NF
CSAR_4 = dT4*4.186e3/concentracion_NF

fig,(ax,ax2,ax3)=plt.subplots(3,1,figsize=(10,12),constrained_layout=True,sharex=True) 
ax.plot(t_1,T_1,'.-',label=paths_NF[0])
ax.plot(t_2,T_2,'.-',label=paths_NF[2])
ax.plot(t_3,T_3,'.-',label=paths_NF[4])
ax.plot(t_4,T_4,'.-',label=paths_NF[6])

ax2.plot(t_1,dT1,'.-',label=paths_NF[0])
ax2.plot(t_2,dT2,'.-',label=paths_NF[2])
ax2.plot(t_3,dT3,'.-',label=paths_NF[4])
ax2.plot(t_4,dT4,'.-',label=paths_NF[6])

ax3.plot(t_1,CSAR_1,'.-',label=paths_NF[0])
ax3.plot(t_2,CSAR_2,'.-',label=paths_NF[2])
ax3.plot(t_3,CSAR_3,'.-',label=paths_NF[4])
ax3.plot(t_4,CSAR_4,'.-',label=paths_NF[6])

for a in [ax,ax2,ax3]:
    a.grid()
    a.legend()
    a.set_xlim(0,)
    #.set_ylim(20,43)
ax.set_ylabel('T (°C)')
ax2.set_ylabel('dT/dt (°C/s)')
ax3.set_ylabel('CSAR (W/g)')
ax3.set_xlabel('t (s)')
plt.suptitle('CSAR - NE@citrico - 15.0 g/L',fontsize=16)
plt.savefig('CSAR_NE.png', dpi=300)

#%%

from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np

dt = t_1[1] - t_1[0]

window = 31    # probar 21–41
poly = 3

# Derivada suavizada
dTdt_sg = savgol_filter(T_1, window_length=window,
                        polyorder=poly,
                        deriv=1,
                        delta=dt)

# Gradiente crudo (para comparar)
dTdt_grad_1 = np.gradient(T_1, dt)
dTdt_grad_2 = np.gradient(T_2, dt)
dTdt_grad_3 = np.gradient(T_3, dt)
#%%
%matplotlib
fig,(ax,ax2)=plt.subplots(2,1,figsize=(8,7),constrained_layout=True,sharex=True)

ax.plot(t_1, T_1, '.-', label='Orig')
ax.plot(t_2, T_2, '.-', label='Orig')
ax.plot(t_3, T_3, '.-', label='Orig')
ax2.plot(t_1, dTdt_grad_1, '.-', label='gradient (crudo)')
ax2.plot(t_2, dTdt_grad_2, '.-', label='gradient (crudo)')
ax2.plot(t_3, dTdt_grad_3, '.-', label='gradient (crudo)')

#ax2.plot(t_1, dTdt_sg, '-', linewidth=2, label='Savitzky-Golay')
ax2.axhline(0, color='k', ls='--')
ax2.set_xlabel('t (s)')
ax2.set_ylabel('dT/dt (°C/s)')

for a in [ax,ax2]:
    a.grid()
    a.legend()
plt.show()



#%%

fig, (ax,ax1,ax2)=plt.subplots(3,1,figsize=(10,8),constrained_layout=True,sharex=True,sharey=Tue)

ax.set_title('300 kHz',loc='left')

ax.plot(t_NE_300_150,T_NE_300_150,'o',label='300_150')
ax.plot(t_NE_300_150[indx_min[0]:],T_NE_300_150[indx_min[0]:],'.',label='300_150')

ax.plot(t_NE_300_100,T_NE_300_100,'.-',label='300_100')

ax.plot(t_NE_300_050,T_NE_300_050,'.-',label='300_050')

ax1.set_title('212 kHz',loc='left')

ax1.plot(t_NE_212_150,T_NE_212_150,label='212_150')
ax1.plot(t_NE_212_100,T_NE_212_100,label='212_100')
#ax1.plot(t_NE_212_050,T_NE_212_050,label='212_050')

ax2.set_title('135 kHz',loc='left')
ax2.plot(t_NE_135_150,T_NE_135_150,'o',label='135_150')    
ax2.plot(t_NE_135_150[indx_min[6]:],T_NE_135_150[indx_min[6]:],'.',label='135_150')    

for a in (ax,ax1,ax2):
    a.grid()
    a.legend()
    a.set_xlim(0,)

plt.suptitle('NE@citrico - coprecipitacion',fontsize=16)
plt.show()

#%%
def ajustes_lineal_T_arbitraria(Tcentral, t, T, label, x=1.0):
    """
    Realiza ajustes lineal alrededor de Tcentral ± x usando curve_fit.
    
    Args:
        Tcentral (float): Temperatura de equilibrio
        t (np.array): Array de tiempos
        T (np.array): Array de temperaturas
        x (float): Rango alrededor de Tcentral (default=1.0)
        
    Returns:
        tuple: (dict_lin, dict_exp) donde:
            - dict_lin: Diccionario con resultados del ajuste lineal
    """
    # Definir la función lineal para curve_fit
    def linear_func(x, a, b):
        return a * x + b
    
    # Crear máscara para el intervalo de interés
    mask = (T >= Tcentral - x) & (T <= Tcentral + x)
    t_interval = t[mask]
    T_interval = T[mask]
    
    # Ajuste lineal con curve_fit
    popt, pcov = curve_fit(linear_func, t_interval, T_interval)
    perr = np.sqrt(np.diag(pcov))  # Desviaciones estándar de los parámetros
    
    # Crear función de ajuste
    poly_lin = lambda x: linear_func(x, *popt)
    
    # Calcular R²
    residuals = T_interval - poly_lin(t_interval)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((T_interval - np.mean(T_interval))**2)
    r2_lin = 1 - (ss_res / ss_tot)
    
    t_fine = np.linspace(t_interval.min()-80, t_interval.max()+80, 100)
    
    # Crear ufloat para la pendiente con su incertidumbre
    pendiente_ufloat = ufloat(popt[0], perr[0])
    
    # Preparar diccionario para resultados lineales
    dict_lin = {
        'pendiente': pendiente_ufloat,
        'ordenada': ufloat(popt[1], perr[1]),
        'r2': r2_lin,
        't_interval': t_interval,
        'T_interval': T_interval,
        'funcion': poly_lin,
        'ecuacion': f"({popt[0]:.3f}±{perr[0]:.3f})t + ({popt[1]:.3f}±{perr[1]:.3f})",
        'rango_x': x,
        'AL_t': t_fine,
        'AL_T': poly_lin(t_fine),
        'covarianza': pcov
    }
    
    # Crear figura 
    fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
    ax.plot(t, T, '.-', label=label)
    
    # Plotear ajustes con el rango extendido que definiste
    ax.plot(t_fine, poly_lin(t_fine), '-', c='tab:green', lw=2, 
            label=f'Ajuste lineal: {dict_lin["ecuacion"]} (R²={r2_lin:.3f})')

    ax.axhspan(Tcentral-x, Tcentral+x, 0, 1, color='tab:red', alpha=0.3, 
               label='T$_{eq}\pm\Delta T$ ='+ f' {Tcentral} $\pm$ {x} ºC')
    
    ax.set_xlabel('t (s)')
    ax.set_ylabel('T (°C)')
    ax.grid()
    ax.legend()
    plt.show()

    # Imprimir resultados (manteniendo tu formato)
    print("\nResultados del ajuste lineal:")
    print(f"Pendiente: {dict_lin['pendiente']} °C/s")
    print(f"Ordenada: {dict_lin['ordenada']} °C")
    print(f"Coeficiente R²: {dict_lin['r2']:.5f}")
    
    
    return dict_lin
#%%# Resultados 
# resultados_FF1 = ajustes_lineal_T_arbitraria(23.0, t_FF1_0, T_FF1,'FF1', x=2)
# resultados_FF2 = ajustes_lineal_T_arbitraria(24.0, t_FF2_0, T_FF2,'FF2', x=5)

resultados_NE_300_150 = ajustes_lineal_T_arbitraria(29.7, t_NE_300_150, T_NE_300_150,'NE 300_150', x=2.5)
resultados_NE_300_100 = ajustes_lineal_T_arbitraria(29.8, t_NE_300_100, T_NE_300_100,'NE 300_100', x=2.5)





#%%
concentracion=ufloat(11.3,0.4)
dTdt_lineal_promedio=np.mean([resultados_FF1['pendiente'],resultados_FF2['pendiente']])
print(f'Pendiente promedio = {dTdt_lineal_promedio:.5f} ºC/s')
CSAR_lineal = dTdt_lineal_promedio*4.186e3/concentracion
print(f'CSAR = {CSAR_lineal:.0f} W/g (ajuste lineal)')


# %%
