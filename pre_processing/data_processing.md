## How to convert CasJobs SQL tables into TFRecord files ##

**1)** Donwload the relevant SQL table from the CasJobs online tool. This can also be done directly in the several SDSS data releases (DR) websites, but it tends to be slower and its output is capped at 500000 rows. We here show an example of one of the queries we used in CasJobs: 

```sql
SELECT p.objid, p.ra, p.dec, p.u, p.g, p.r, p.i, p.z,
  s.z as redshift, s.plate, s.mjd, s.fiberid 
FROM PhotoObj AS p INNER JOIN SpecObj AS s ON s.bestobjid = p.objid
WHERE 
  s.class = 'GALAXY' AND (s.subclass = 'STARFORMING' OR s.subclass = 'STARBURST')
```

**2)** The SQL tables do not contain any spectra, only their details. In order to get access to the spectra, we need to get their web paths first. We thus run:

```bash
python spectra_get_paths.py
```

making sure that the path to the downloaded SQL table is correct. The spectra naming conventions may change for different DRs or for different objects (galaxies, stars, ...), so one has also to make sure that the FITS path is right. To check online if the spectra paths exist, one can access 'https://dr15.sdss.org/sas/dr15/' (for the case of DR15) followed by the path produced by the ```spectra_get_paths.py``` macro. Note that the path to be used by ```rsync``` is slightly different. Look online for 'sdss dr15 download' to have access to all the relevant information. 

**3)** Once the paths are obtained the spectra can be downloaded. Run the following:

```bash
bash spectra_download.sh
```

but make sure there is enough free disk space in the target folder. This step is the one that takes the longest to finish. It is recommended to run it first using the ```rsync --dry-run``` option. One can alternatively using the ```wget``` tool. If the transfer is estimated to involve more than 1TB, one should not use these tools, but contact the SDSS helpdesk: helpdesk@sdss.org.

**3b)** After downloading the data, it will be stored under an unnecessary number of subfolders. It is probably a good idea to use ```mv``` to move the FITS files to the head directory defined in the ```spectra_download.sh``` macro, and then use ```rmdir``` to remove the empty directories left.


**4)** Once the spectral data has been downloaded, it is useful to store some parameters along with their local path. For instance, when one wants to deredshift a spectra, one needs to know the redshift of each spectra. This information is not provided in the 'lite' version of the SDSS spectra. If we run:

```bash
python spectra_list_values.py
```

a new file will be created with all the relevant information. Do not forget to check all the paths being used.

**5)** At this point we have all information needed to store the data in TFRecords format. We use the ```real_spectra_to_tfrecord``` function stored in ```src/data/tfrecords.py``` to make the conversion. By slightly changing the function, the user can select which of the parameters saved with ```spectra_list_values.py``` wants to convert to TFRecord, or even using them directly in the spectra.

* Why using the TFRecord file format at all? It is the recommended format when using Tensorflow, irrespective of the data being used.

**6)** The data can then be feeded in the GAN for training, by changing some constructor arguments. See ```spectra_dcgan.py``` for a good example. Reading the documentation of the ```python _load_data()``` method may also help. The GAN model is stored under ```src/models/gans.py```.

