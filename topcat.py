
import matplotlib.pyplot as plt
import sys
import numpy as np
import h5py as h5
from astropy.samp import SAMPIntegratedClient, SAMPHubError
from astropy.table import Table
import os
from time import sleep

client = SAMPIntegratedClient(
    name="SDSS-V Astra Spectrum Viewer",
    description="Visualize SDSS-V spectra",
    metadata={
        "samp.icon.url": "https://www.sdss.org/wp-content/uploads/2022/04/cropped-cropped-site_icon_SDSSV-32x32.png"
    }
)
try:
    client.connect()
except SAMPHubError:
    os.system("open /Applications/TOPCAT.app")
    for i in range(5):
        sleep(1)
        try:
            client.connect()
        except SAMPHubError:
            continue
        else:
            break
    else:
        raise

fig, (ax_spectrum, ax_rectified_spectrum) = plt.subplots(2, 1, figsize=(20, 6), gridspec_kw={'height_ratios': [3, 1]})
λ = 10**(4.179 + 6e-6 * np.arange(8575))[:-1]

plot_data, = ax_spectrum.plot(λ, np.nan * np.ones_like(λ), lw=2, color='black', label='Spectrum')
plot_model, = ax_spectrum.plot(λ, np.nan * np.ones_like(λ), lw=2, color="tab:red")
plot_initial_model, = ax_spectrum.plot(λ, np.nan * np.ones_like(λ), lw=1, c="#666666", zorder=-1)

plot_rectified_data, = ax_rectified_spectrum.plot(λ, np.nan * np.ones_like(λ), lw=2, color='black', label='Spectrum')
plot_rectified_model, = ax_rectified_spectrum.plot(λ, np.nan * np.ones_like(λ), lw=2, color="tab:red")
plot_rectified_initial_model, = ax_rectified_spectrum.plot(λ, np.nan * np.ones_like(λ), lw=1, c="#666666", zorder=-1)


input_path = sys.argv[1]
#input_path = "20250701_cluster_results.h5"


class Receiver:
    def __init__(self, client):
        self.client = client
        self.received = False
        self.fp = h5.File(input_path, "r")
    def receive_call(self, private_key, sender_id, msg_id, mtype, params, extra):
        self.params = params
        self.received = True
        self.client.reply(msg_id, {"samp.status": "samp.ok", "samp.result": {}})
        print(f"Received call: {mtype} with params: {params}, sender_id: {sender_id}, msg_id: {msg_id}, private_key: {private_key}, extra: {extra}")

        if mtype == "table.highlight.row":
            index = int(params["row"])
            flux = self.fp["flux"][index]

            plot_data.set_ydata(flux)
            plot_model.set_ydata(self.fp["model_flux"][index])
            plot_initial_model.set_ydata(self.fp["initial_model_flux"][index] * self.fp["initial_continuum"][index])

            plot_rectified_data.set_ydata(flux / self.fp["continuum"][index])
            plot_rectified_model.set_ydata(self.fp["model_flux"][index] / self.fp["continuum"][index])
            plot_rectified_initial_model.set_ydata(self.fp["initial_model_flux"][index])

            print(f"Highlighting row {index + 1} with flux data.")
            ax_spectrum.set_title(self.fp["sdss_id"][index])
            for ax in (ax_spectrum, ax_rectified_spectrum):
                ax.set_xlim(λ[0], λ[-1])
            
            ax_rectified_spectrum.set_ylim(0, 1.2)
            edge = 0.05 * np.ptp(flux[(flux > 0) * np.isfinite(flux)])

            ax_spectrum.set_ylim(
                np.max([0, np.nanmin(flux) - edge]),
                np.nanmax(flux) + edge
            )

            fig.tight_layout()
            plt.draw()

    def receive_notification(self, private_key, sender_id, mtype, params, extra):
        self.params = params
        self.received = True
        print(f"Received notification: {mtype} with params: {params}, sender_id: {sender_id}, private_key: {private_key}, extra: {extra}")  

r = Receiver(client)

client.bind_receive_call("table.load.votable", r.receive_call)
client.bind_receive_notification("table.load.votable", r.receive_notification)
client.bind_receive_call("table.highlight.row", r.receive_call)
client.bind_receive_notification("table.highlight.row", r.receive_notification)

print(client.get_registered_clients())

# Send the results table to topcat.

fp = h5.File(input_path, "r")
results_path = os.path.abspath(f"{input_path}.vot")
keys = fp["results"].keys()
data = {key: fp["results"][key] for key in keys}

chi2 = (fp["flux"][:] - fp["model_flux"][:])**2 * fp["ivar"]
chi2 = np.sum(chi2[:, 7287:7934], axis=1) / (7934-7287)

data["rchi2_16750"] = chi2 
t = Table(data)
t.write(results_path, overwrite=True, format='votable')

# Insert the PARAMs stuff
with open(results_path, "r") as fp:
    contents = fp.readlines()
    for params_index, line in enumerate(contents):
        if line.strip() == "<TABLE>":
            break
    else:
        raise ValueError("Could not find <TABLE> tag in the VOTable file.")

n_columns = len(data.keys())
col_specs = f"".join([f"col:{i: <2d}" for i in range(n_columns)])
assert n_columns < 100 # otherwise need to change the 6x{n_columns} below 
column_visibilities = " ".join(["true"] * n_columns)
column_indices = " ".join([str(i) for i in range(n_columns)])

subset_names_and_specs = [
    ("All", "all"),
    ("Activated", "activated:5872"), # TODO wtf is 5872? index i activated before i saved?
]
for i, name in enumerate(t.columns.keys()):
    if isinstance(t[name][0], np.bool):
        subset_names_and_specs.append((name, f"bcol:{i}"))
max_len = max([max([len(name), len(spec)]) for name, spec in subset_names_and_specs])
n_subsets = len(subset_names_and_specs)

subset_names = f"".join([name.ljust(max_len) for name, _ in subset_names_and_specs])
subset_specs = f"".join([spec.ljust(max_len) for _, spec in subset_names_and_specs])

params_string = f"""
    <PARAM arraysize="3" datatype="char" name="TC_codecVersion" utype="topcat_session:codecVersion" value="2.0"/>
    <PARAM arraysize="6" datatype="char" name="TC_topcatVersion" utype="topcat_session:topcatVersion" value="4.10-1"/>
    <PARAM arraysize="{len(input_path) + 4}" datatype="char" name="TC_label" utype="topcat_session:label" value="{input_path}.vot"/>
    <PARAM arraysize="6x{n_columns}" datatype="char" name="TC_colSpecs" utype="topcat_session:colSpecs" value="{col_specs}"/>
    <PARAM arraysize="{max_len}x{n_subsets}" datatype="char" name="TC_subsetNames" utype="topcat_session:subsetNames" value="{subset_names}"/>
    <PARAM arraysize="{max_len}x{n_subsets}" datatype="char" name="TC_subsetSpecs" utype="topcat_session:subsetSpecs" value="{subset_specs}"/>
    <PARAM arraysize="{n_columns}" datatype="int" name="TC_columnIndices" utype="topcat_session:columnIndices" value="{column_indices}"/>
    <PARAM arraysize="{n_columns}" datatype="boolean" name="TC_columnVisibilities" utype="topcat_session:columnVisibilities" value="{column_visibilities}"/>
    <PARAM datatype="int" name="TC_currentSubset" utype="topcat_session:currentSubset" value="0"/>
    <PARAM arraysize="1387" datatype="char" name="TC_activationActions" utype="topcat_session:activationActions" value='[
  {{"_ActivationType":"uk.ac.starlink.topcat.activate.TopcatSkyPosActivationType","ra_col":"","ra_unit":"degree","dec_col":"","dec_unit":"degree"}},
  {{"_ActivationType":"uk.ac.starlink.topcat.activate.SendSkyPosActivationType","ra_col":"","ra_unit":"degree","dec_col":"","dec_unit":"degree"}},
  {{"_ActivationType":"uk.ac.starlink.topcat.activate.ViewHips2fitsActivationType","ra_col":"","ra_unit":"degree","dec_col":"","dec_unit":"degree","fov_col":"1.0","fov_unit":"degree","hips":"DSS2/color","npix":"300"}},
  {{"_ActivationType":"uk.ac.starlink.topcat.activate.SendHips2fitsActivationType","ra_col":"","ra_unit":"degree","dec_col":"","dec_unit":"degree","fov_col":"1.0","fov_unit":"degree","hips":"DSS2/red","npix":"300"}},
  {{"_ActivationType":"uk.ac.starlink.topcat.activate.DelayActivationType","seconds":"1"}},
  {{"_ActivationType":"uk.ac.starlink.topcat.activate.JelActivationType","text":"","sync":"false"}},
  {{"_ActivationType":"uk.ac.starlink.topcat.activate.ShellActivationType","word0":"","nword":"5","word1":"","word2":"","word3":"","word4":"","sync":"false","capture":"true"}},
  {{"_ActivationType":"uk.ac.starlink.topcat.activate.SendCustomActivationType","mtype":"","nparam":"3","pname0":"","pexpr0":"","pname1":"","pexpr1":"","pname2":"","pexpr2":""}},
  {{"_ActivationType":"uk.ac.starlink.topcat.activate.SendIndexActivationType","_isActive":"true","_isSelected":"true"}}
]'/>
"""
contents = "".join(contents[:params_index]) + " <TABLE>\n    " + params_string.lstrip() + "".join(contents[params_index + 1:])
with open(results_path, "w") as fp:
    fp.write(contents)

print(f"Sending file: {results_path}")
client.notify_all({
    "samp.mtype": "table.load.votable",
    "samp.params": {
        "url": f"file://{results_path}",
    }
})

ax_rectified_spectrum.set_xlabel(r"Wavelength [$\AA$]")
ax_spectrum.set_ylabel(r"Flux")   
fig.tight_layout()

try:
    plt.show()
except KeyboardInterrupt:
    client.disconnect()

