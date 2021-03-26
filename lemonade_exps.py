from lemonade.setup import *
from fastai.imports import *

print(DEVICE)
print(DATA_STORE)

from lemonade.data import *
ehr_1K_data = EHRData(PATH_1K, LABELS)

from lemonade.preprocessing.vocab import *
demograph_dims, rec_dims, demograph_dims_wd, rec_dims_wd = get_all_emb_dims(EhrVocabList.load(PATH_1K))
train_dl, valid_dl, train_pos_wts, valid_pos_wts = ehr_1K_data.get_data()

from lemonade.learn import *
train_loss_fn, valid_loss_fn = get_loss_fn(train_pos_wts), get_loss_fn(valid_pos_wts)

from lemonade.models import *
model = EHR_LSTM(demograph_dims, rec_dims, demograph_dims_wd, rec_dims_wd).to(DEVICE)
optimizer = torch.optim.Adagrad(model.parameters())
h = RunHistory(LABELS)

from lemonade.metrics import *
print(train_dl.dataset.x[1].age_now)

h = fit(5, h, model, train_loss_fn, valid_loss_fn, optimizer, auroc_score, \
              train_dl, valid_dl, to_chkpt_path=None, from_chkpt_path=None, verbosity=1)