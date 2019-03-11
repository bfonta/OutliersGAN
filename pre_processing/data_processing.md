## How to convert CasJobs SQL tables into TFRecord files ##

Steps

* **1)** Donwload the relevant SQL table from the CasJobs online tool. This can also be done directly in the several SDSS data releases (DR) websites, but it tends to be slower and its output is capped at 500000 rows. We here show an example of one of the queries we used in CasJobs: 

```sql
SELECT p.objid, p.ra, p.dec, p.u, p.g, p.r, p.i, p.z,
  s.z as redshift, s.plate, s.mjd, s.fiberid into mydb.gal_starforming_starburst from PhotoObj AS p INNER JOIN SpecObj AS s ON s.bestobjid = p.objid
WHERE 
  s.class = 'GALAXY' AND (s.subclass = 'STARFORMING' OR s.subclass = 'STARBURST')
```

* **2)** The SQl tables do not contain any spectra, but their details only. In order to get access to the spectra, we need to get their web paths first. I thus run

```bash
python spectra_get_paths.py
```

making sure that the path to the downloaded SQL table is correct. The spectra naming conventions may change for different data releases or for different objects, so one has also to make sure that the fits path is right. For checking online if the spectra paths exist, one can enter 'https://dr15.sdss.org/sas/dr15/' (for the case of DR15) followed by the path produced by the spectra_get_paths.py macro. Note that the path to be used by rsync is slightly different. Looking online for 'sdss dr15 download' provides all the relevant information. 

* **3)** Once the paths are obtained the spectra can be downloaded. Run

```bash
bash spectra_download.sh
```

but make sure there is enough free disk space in the target folder. This step is the one that takes the longest to finish. It is recommended to run it first using the rsync '--dry-run' option.
