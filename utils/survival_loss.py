import torch


def cox_ph_loss(risk, time, event):
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    event = event[order]

    hazard_ratio = torch.exp(risk)
    log_cum_hazard = torch.log(torch.cumsum(hazard_ratio, dim=0))
    log_risk = risk

    diff = log_risk - log_cum_hazard
    loss = -torch.sum(diff * event) / (event.sum() + 1e-8)
    return loss
