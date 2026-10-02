import pickle
import pandas as pd


# ELEMENT_LIST = ['Pt', 'Ru', 'Ir', 'Ni', 'Cu', 'Pd', 'Co', 'Rh', 'Fe', 'Ag', 'Au']
ELEMENT_LIST = ['V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ru', 'Rh', 'Pd', 'Ag', 'Ir', 'Pt', 'Au']


def extract_bulk(ele_list, bulk_db_path):
    bulk_db = pickle.load(open(bulk_db_path, "rb"))

    bulk_symb_list = []
    bulk_mpid_list = []
    for i in range(len(bulk_db)):
        bulk_obj = bulk_db[i]
        atoms, src_id = bulk_obj["atoms"], bulk_obj["src_id"]
        bulk_ele = list(set(atoms.get_chemical_symbols()))
        select = True
        for _ele in bulk_ele:
            if _ele not in ele_list:
                select = False
                break
        if select:
            bulk_symb_list.append(atoms.symbols)
            bulk_mpid_list.append(src_id)

    return bulk_symb_list, bulk_mpid_list


bulk_symb_list, bulk_mpid_list = extract_bulk(ELEMENT_LIST, 'pkls/bulks.pkl')

print(len(bulk_symb_list))

df_dict = {
    'bulk_symbols': bulk_symb_list,
    'bulk_mpid': bulk_mpid_list,
}
df = pd.DataFrame.from_dict(df_dict)
df.to_csv('select_bulks.csv', index=False)
