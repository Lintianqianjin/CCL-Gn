# CCL-Gn
Official code of "Unraveling and Mitigating Endogenous Task-oriented Spurious Correlations in Ego-graphs via Automated Counterfactual Contrastive Learning"

---
An example to run the experiments on GOOD-WebKB dataset with Concept shift.
```python
python CCLGn.py --encoder GCNGOOD --task WebKBConcept --lr 0.001 --hid_dim 300 --batch_size 200 --accumulation_steps 1 --min_step 200 --max_step 1000 --patience 50 --device 0 --num_hops 3 --size_reg 1e-05 --rank_coef 1.0 --cont_temp 0.05 --ood
```