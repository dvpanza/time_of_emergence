#%%
import iris 
import datetime as dt
import matplotlib.pyplot as plt
from tae import rolling_ks_2samp, get_SN_ratio, get_tae_SN, get_STD, get_LF

#Load data
data = iris.load("data/tas_DJF_ACCESS-CM2_r1i1p1f1.nc")[0]

#Preindustrial subset
start_date = dt.datetime(1850, 1, 1)
end_date = dt.datetime(1910, 12, 31)
time_constraint = iris.Constraint(time=lambda cell: start_date <= cell.point <= end_date)
preind = data.extract(time_constraint)

#Industrial subset
start_date = dt.datetime(1911, 1, 1)
end_date = dt.datetime(2100, 12, 31)
time_constraint = iris.Constraint(time=lambda cell: start_date <= cell.point <= end_date)
ind = data.extract(time_constraint)

#Emergence with KS
tae_ks = rolling_ks_2samp(preind, ind, window=20, step=5)
print(f"TAE with KS: {tae_ks}")


#Emergence with SN

# Obtener la coordenada de tiempo
preind_time = preind.coord('time')
ind_time = ind.coord('time')

# Convertir los puntos (días desde 1850) a fechas reales
dates_preind = preind_time.units.num2date(preind_time.points)
dates_ind = ind_time.units.num2date(ind_time.points)

#Extraer los años
years_preind = [date.year for date in dates_preind]
years_ind = [date.year for date in dates_ind]

#Obtener datos de la serie de tiempo
values_preind = preind.data
mean = values_preind.mean()

values_ind = ind.data - mean
values_preind = values_preind - mean

sn = get_SN_ratio(
    years_ind, 
    values_ind, 
    years_preind, 
    values_preind, 
    method="lowess", 
    frac=0.1,
    )

tae_sn = get_tae_SN(sn, years_ind, sn_threshold=1)
print(f"TAE with SN: {tae_sn}")

#%% Visualization

LF = get_LF(years_ind, values_ind, method="lowess", frac=0.1)
STD = get_STD(years_preind,values_preind)

fig,axes = plt.subplots(2,1, figsize=(10, 6), sharex=True)

ax = axes[0]
ax.plot(years_ind, values_ind, color='tab:blue')
ax.plot(years_preind, values_preind, color='tab:green')
ax.axvline(tae_ks, color='tab:orange', linestyle='--', label=f"TAE with KS: {tae_ks}")
ax.axvline(tae_sn, color='tab:red', linestyle='--', label=f"TAE with SN: {tae_sn}")
ax.plot(years_ind, LF, color='tab:purple', label="LF component")
ax.axhline(y=STD, color='tab:grey', linestyle='--', label="STD")
ax.axhline(y=-STD, color='tab:grey', linestyle='--')
ax.grid(True)
ax.set_ylabel("°C")
ax.legend()

ax = axes[1]
ax.plot(years_ind, sn, color='tab:red', label="SN")
ax.axhline(y=1, color='k', linestyle='-')
ax.axvline(tae_sn, color='tab:red', linestyle='--', label=f"TAE with SN: {tae_sn}")
ax.legend()
ax.grid(True)
ax.set_xlabel("Year")  
ax.set_ylabel("SN")

plt.subplots_adjust(hspace=0.1)


# %%
