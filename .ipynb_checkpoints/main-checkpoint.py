import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# import wandb
import numpy as np
import torch
import torch.nn as nn
import pickle
import xml.etree.ElementTree as ET
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score, 
    matthews_corrcoef
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from accelerate import Accelerator
# Imports specific to the custom peft lora model
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType

