import os

import pandas as pd

from Dataset import constants


def compare_unawareness_unawarenessHbc_summary(file_path,target_acc,sensitive_acc,fair_acc):

    def generate_columns():

        def generate_model_parameter_columns():
            model_parameter_cols = ['K_Y','K_S','B_S']
            return model_parameter_cols

        def generate_acc_columns():
            acc_cols = ['Y_UNA_ACC','Y_UNAHBC_ACC','S_UNA_ACC','S_UNAHBC_ACC']
            return acc_cols

        def generate_fair_columns():
            fair_cols = ['EFG_UNA','EFG_UNAHBC']
            return  fair_cols

        cols = generate_model_parameter_columns()
        cols += generate_acc_columns()
        cols += generate_fair_columns()
        return cols

    def generate_row():
        data = [constants.K_Y,constants.K_S,constants.BETA_S,
                *target_acc,*sensitive_acc,*fair_acc]
        return data

    mode = 'a' if os.path.isfile(file_path) else 'w'
    columns = generate_columns()
    row = generate_row()
    df = pd.DataFrame([row], columns=columns)
    constants.log.info(f"Writing summary:\n{df}")
    df.to_csv(file_path, mode=mode, float_format="%.4f", header=mode == 'w', index=False)

def compare_unawareness_unawarenessHbc_entropy(file_path,target_acc,sensitive_acc,entropy):
    def generate_columns():

        def generate_model_parameter_columns():
            model_parameter_cols = ['K_Y','K_S','B_S']
            return model_parameter_cols

        def generate_acc_columns():
            acc_cols = ['Y_UNA_ACC','Y_UNAHBC_ACC','S_UNA_ACC','S_UNAHBC_ACC']
            return acc_cols

        def generate_fair_columns():
            fair_cols = ['ENTROPY_UNA','ENTROPY_UNAHBC']
            return  fair_cols

        cols = generate_model_parameter_columns()
        cols += generate_acc_columns()
        cols += generate_fair_columns()
        return cols

    def generate_row():
        data = [constants.K_Y,constants.K_S,constants.BETA_S,
                *target_acc,*sensitive_acc,*entropy]
        return data

    mode = 'a' if os.path.isfile(file_path) else 'w'
    columns = generate_columns()
    row = generate_row()
    df = pd.DataFrame([row], columns=columns)
    constants.log.info(f"Writing summary:\n{df}")
    df.to_csv(file_path, mode=mode, float_format="%.4f", header=mode == 'w', index=False)

def compare_hbc_unahbc_summary(file_path,target_acc,sensitive_acc,fair_acc):

    def generate_columns():

        def generate_model_parameter_columns():
            model_parameter_cols = ['K_Y','K_S','B_S']
            return model_parameter_cols

        def generate_acc_columns():
            acc_cols = ['Y_HBC_ACC','Y_UNAHBC_ACC','S_HBC_ACC','S_UNAHBC_ACC']
            return acc_cols

        def generate_fair_columns():
            fair_cols = ['EFG_HBC','EFG_UNAHBC']
            return  fair_cols

        cols = generate_model_parameter_columns()
        cols += generate_acc_columns()
        cols += generate_fair_columns()
        return cols

    def generate_row():
        data = [constants.K_Y,constants.K_S,constants.BETA_S
                *target_acc,*sensitive_acc,*fair_acc]
        return data

    mode = 'a' if os.path.isfile(file_path) else 'w'
    columns = generate_columns()
    row = generate_row()
    df = pd.DataFrame([row], columns=columns)
    constants.log.info(f"Writing summary:\n{df}")
    df.to_csv(file_path, mode=mode, float_format="%.4f", header=mode == 'w', index=False)

def compare_decode_paramG(file_path,rows,columns):
    mode = 'a' if os.path.isfile(file_path) else 'w'
    df = pd.DataFrame([rows], columns=columns)
    constants.log.info(f"Writing summary:\n{df}")
    df.to_csv(file_path, mode=mode, float_format="%.4f", header=mode == 'w', index=False)

def write_experiment_summary(args, file_path,train_acc,test_acc,fair_acc):

    def generate_columns():

        def generate_model_parameter_columns():
            model_parameter_cols = ['K_Y','K_S','B_X','B_Y','B_S']
            return model_parameter_cols

        def generate_acc_columns():
            acc_cols = ['Train_T_ACC','Train_S_ACC','Test_T_ACC','Test_S_ACC']
            return acc_cols

        def generate_fair_columns():
            fair_cols = ['Fair_epsilon','Fair_delta','Mg','EFG','Fair_threshold','Pass']
            return  fair_cols

        cols = ['run_name', 'dataset_name', 'modelF', 'modelG', 'lr', 'epoch']
        cols += generate_model_parameter_columns()
        cols += generate_acc_columns()
        cols += generate_fair_columns()
        return cols

    def generate_row(args,train_acc,test_acc,fair_acc):
        data = [args.run_name, constants.DATASET, "default","default",
                args.server_lr, args.server_epochs,constants.K_Y,constants.K_S,
                constants.BETA_X,constants.BETA_Y,constants.BETA_S,*train_acc,
                *test_acc,*fair_acc]
        return data

    file_path = os.path.join(file_path, constants.summary_file)
    mode = 'a' if os.path.isfile(file_path) else 'w'
    columns = generate_columns()
    row = generate_row(args,train_acc,test_acc,fair_acc)
    df = pd.DataFrame([row], columns=columns)
    constants.log.info(f"Writing summary:\n{df}")
    df.to_csv(file_path, mode=mode, float_format="%.4f", header=mode == 'w', index=False)
