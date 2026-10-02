import pickle
import pandas as pd
from mp_api.client import MPRester


ELEMENT_LIST = ['V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ru', 'Rh', 'Pd', 'Ag', 'Ir', 'Pt', 'Au']
YOUR_MP_API="VeAe0zjm0alYpC9cj2Kz8j0KoVNtVbKo"

with MPRester(YOUR_MP_API) as mpr:
    docs = mpr.summary.search(
       fields=["material_id", "theoretical", "structure"]
    )
    stable_maters = [
        (doc.material_id, doc.structure.to_ase_atoms())
        for doc in docs if doc.theoretical is False
    ]

# with MPRester(YOUR_MP_API) as mpr:
#     docs = mpr.summary.search(
#        is_stable=True, fields=["material_id", "structure"]
#     )
#     stable_maters = [
#         (doc.material_id, doc.structure.to_ase_atoms())
#         for doc in docs
#     ]

print(len(stable_maters))
with open('pkls/exp_mater_from_mp.pkl', 'wb') as f:
    pickle.dump(stable_maters, f)

bulk_symb_list = []
bulk_mpid_list = []
for mpid, atoms in stable_maters:
    bulk_ele = list(set(atoms.get_chemical_symbols()))
    select = True
    for _ele in bulk_ele:
        if _ele not in ELEMENT_LIST:
            select = False
            break
    if select:
        bulk_symb_list.append(atoms.symbols)
        bulk_mpid_list.append(mpid)


df_dict = {
    'bulk_symbols': bulk_symb_list,
    'bulk_mpid': bulk_mpid_list,
}
df = pd.DataFrame.from_dict(df_dict)
df.to_csv('select_bulk_mp3.csv', index=False)
