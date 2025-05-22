import os
import shutil
import pandas as pd
from register import Register
import sys  # ja el tens si no, afegeix-lo

# Redirecció de sortida
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

BASE_IMATGES_PATH = "/mnt/work/users/josep/WSI/"
BASE_RESULTATS_PATH = "/mnt/work/users/albert.caus/test/Sequencial/"
BASE_TXT_PATH = "/home/usuaris/imatge/albert.caus/test/Exemples/Sequencial/"
TIPUS_REGISTRE = "normal"

def llegir_llista_imatges(nom_fitxer):
    with open(nom_fitxer, 'r') as f:
        return [linia.strip().replace('"', '') for linia in f.readlines() if linia.strip()]

def construir_path_complet(nom_imatge):
    prefix = nom_imatge[:11]
    return os.path.join(BASE_IMATGES_PATH, prefix, nom_imatge + ".mrxs")

def arredoneix(valor):
    return round(valor, 4) if isinstance(valor, (int, float)) else valor

def processar_cas(nom_cas):
    fitxer_he = os.path.join(BASE_TXT_PATH, f"{nom_cas}_HE.txt")
    fitxer_ihc = os.path.join(BASE_TXT_PATH, f"{nom_cas}_IHC.txt")


    imatges_he = llegir_llista_imatges(fitxer_he)
    imatges_ihc = llegir_llista_imatges(fitxer_ihc)

    metrics = {ihc: {
        "Rigid_IoU": [""] * len(imatges_he),
        "Rigid_Corr": [""] * len(imatges_he),
        "NonRigid_IoU": [""] * len(imatges_he),
        "NonRigid_Corr": [""] * len(imatges_he)
    } for ihc in imatges_ihc}

    directori_cas = os.path.join(BASE_RESULTATS_PATH, nom_cas)
    if os.path.exists(directori_cas):
        shutil.rmtree(directori_cas)
    os.makedirs(directori_cas)


    log_path = os.path.join(directori_cas, "sortida.txt")
    logfile = open(log_path, "w")
    sys.stdout = Tee(sys.__stdout__, logfile)  # Mostra també per consola si cal
    sys.stderr = Tee(sys.__stderr__, logfile)

    for idx_he, imatge_he in enumerate(imatges_he):
        for imatge_ihc in imatges_ihc:
            print(f"[{nom_cas}] Registrant HE: {imatge_he} amb IHC: {imatge_ihc}")

            nom_subcarpeta = f"{imatge_he}__vs__{imatge_ihc}"
            directori_individual = os.path.join(directori_cas, nom_subcarpeta)

            if os.path.exists(directori_individual):
                shutil.rmtree(directori_individual)
            os.makedirs(directori_individual)

            path_he = construir_path_complet(imatge_he)
            path_ihc = construir_path_complet(imatge_ihc)

            imatges_a_registrar = [path_he, path_ihc]
            ref_img = path_ihc

            registre = Register(imatges_a_registrar, directori_individual, TIPUS_REGISTRE, ref_img)

            if TIPUS_REGISTRE == "normal":
                try:
                    iou_rigid, corr_rigid, iou_non_rigid, corr_non_rigid = registre.registration()
                    print(f'IoU_rigid: {iou_rigid}, corr_rigid: {corr_rigid}')
                    print(f'IoU_non_rigid: {iou_non_rigid}, corr_non_rigid: {corr_non_rigid}')

                    metrics[imatge_ihc]["Rigid_IoU"][idx_he] = iou_rigid
                    metrics[imatge_ihc]["Rigid_Corr"][idx_he] = corr_rigid
                    metrics[imatge_ihc]["NonRigid_IoU"][idx_he] = iou_non_rigid
                    metrics[imatge_ihc]["NonRigid_Corr"][idx_he] = corr_non_rigid
                except Exception as e:
                    print(f"[ERROR] Fallida en el registre entre {imatge_he} i {imatge_ihc}: {e}")
            elif TIPUS_REGISTRE == "hd":
                registre.registration_hd()
            else:
                print("Tipus de registre no vàlid!")

    # Crear Excel del cas
    columnes = [("Ref. image:", "", "")]
    valors = []

    for idx, imatge_he in enumerate(imatges_he):
        fila = [imatge_he]
        for ihc in imatges_ihc:
            fila.extend([
                arredoneix(metrics[ihc]["Rigid_IoU"][idx]),
                arredoneix(metrics[ihc]["Rigid_Corr"][idx]),
                arredoneix(metrics[ihc]["NonRigid_IoU"][idx]),
                arredoneix(metrics[ihc]["NonRigid_Corr"][idx]),
            ])
        valors.append(fila)

    for ihc in imatges_ihc:
        columnes.extend([
            (ihc, "Rigid", "IoU"),
            (ihc, "Rigid", "Corr"),
            (ihc, "Non-rigid", "IoU"),
            (ihc, "Non-rigid", "Corr"),
        ])

    multiindex_cols = pd.MultiIndex.from_tuples(columnes)
    df = pd.DataFrame(valors, columns=multiindex_cols)

    excel_path = os.path.join(directori_cas, f"metriques_{nom_cas}.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Full 1")
    print(f"[INFO] Excel guardat per al cas {nom_cas} a: {excel_path}")

def main():
    fitxer_casos = os.path.join(BASE_TXT_PATH, "Estadistiques_sequencials.txt")
    noms_casos = llegir_llista_imatges(fitxer_casos)

    for nom_cas in noms_casos:
        print(f"\n=== PROCESSANT CAS: {nom_cas} ===")
        processar_cas(nom_cas)

if __name__ == "__main__":
    main()


    #ARREGLAR EL PROBLEMA DE QUE SI EXECUTO UN CAS EN UN DIRECTORI QUE JA HI HA ALGO PETA



