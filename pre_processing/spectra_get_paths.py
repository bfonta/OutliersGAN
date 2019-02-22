import pandas as pd

df = pd.read_csv('/fred/oz012/Bruno/data/BOSS_GALCMASS.csv', skiprows=0)
df = df[['plate', 'mjd', 'fiberid']]

path = '/eboss/spectro/redux/v5_10_0/spectra/lite/'
with open('/fred/oz012/Bruno/data/BossGALCMASSList.txt', 'w') as f:
    for i in range(len(df)):
        if i%10000==0: print(i)
        plate = str(df['plate'].iloc[i])
        mjd = str(df['mjd'].iloc[i])
        fiberid = str(df['fiberid'].iloc[i])
        f.write(path +  plate + '/spec-' + plate + '-' + mjd + '-' + fiberid.zfill(4) + '.fits\n')
