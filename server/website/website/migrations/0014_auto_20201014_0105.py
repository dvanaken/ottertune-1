# -*- coding: utf-8 -*-
# Generated by Django 1.11.27 on 2020-10-14 08:05
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('website', '0013_backupdata_other'),
    ]

    operations = [
        migrations.AlterField(
            model_name='session',
            name='hyperparameters',
            field=models.TextField(default='{\n    "DDPG_ACTOR_HIDDEN_SIZES": [128, 128, 64],\n    "DDPG_ACTOR_LEARNING_RATE": 0.02,\n    "DDPG_CRITIC_HIDDEN_SIZES": [64, 128, 64],\n    "DDPG_CRITIC_LEARNING_RATE": 0.001,\n    "DDPG_BATCH_SIZE": 32,\n    "DDPG_GAMMA": 0.0,\n    "DDPG_SIMPLE_REWARD": true,\n    "DDPG_UPDATE_EPOCHS": 30,\n    "DDPG_USE_DEFAULT": false,\n    "DNN_CONTEXT": false,\n    "DNN_DEBUG": true,\n    "DNN_DEBUG_INTERVAL": 100,\n    "DNN_EXPLORE": false,\n    "DNN_EXPLORE_ITER": 500,\n    "DNN_GD_ITER": 100,\n    "DNN_NOISE_SCALE_BEGIN": 0.1,\n    "DNN_NOISE_SCALE_END": 0.0,\n    "DNN_TRAIN_ITER": 100,\n    "FLIP_PROB_DECAY": 0.5,\n    "GPR_BATCH_SIZE": 3000,\n    "GPR_CONTEXT": false,\n    "GPR_DEBUG": true,\n    "GPR_EPS": 0.001,\n    "GPR_EPSILON": 1e-06,\n    "GPR_LEARNING_RATE": 0.01,\n    "GPR_LENGTH_SCALE": 2.0,\n    "GPR_MAGNITUDE": 1.0,\n    "GPR_MAX_ITER": 500,\n    "GPR_MAX_TRAIN_SIZE": 7000,\n    "GPR_MU_MULTIPLIER": 1.0,\n    "GPR_MODEL_NAME": "BasicGP",\n    "GPR_HP_LEARNING_RATE": 0.001,\n    "GPR_HP_MAX_ITER": 5000,\n    "GPR_RIDGE": 1.0,\n    "GPR_SIGMA_MULTIPLIER": 1.0,\n    "GPR_UCB_SCALE": 0.2,\n    "GPR_USE_GPFLOW": true,\n    "GPR_UCB_BETA": "get_beta_td",\n    "IMPORTANT_KNOB_NUMBER": 10000,\n    "INIT_FLIP_PROB": 0.3,\n    "NUM_SAMPLES": 30,\n    "TF_NUM_THREADS": 4,\n    "TOP_NUM_CONFIG": 10}'),
        ),
    ]
