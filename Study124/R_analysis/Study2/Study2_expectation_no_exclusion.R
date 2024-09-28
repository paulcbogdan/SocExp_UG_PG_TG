# --------
library(lme4)
library(lmerTest)
library(dplyr)
library(optimx)

# -----------
library(rstudioapi)
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path))
# --------------

path = r'(..\..\UG_data\processed_RoleChange_Study2.csv)'
df <- read.csv(path)

# ---------
df <- filter(df,
             received < 5 &
             #excluded == 0 &
             id != 39 & id != 88 & # cases of extreme low response rate still excluded
             E_p > 0 & E_r > 0 &
             !is.na(prev_response_bool) & !is.na(subject_response_bool) &
             condition == 'selfish' # can change to: condition == 'replication'
)
# -------------
df$received= scale(df$received)
df$E_p = scale(df$E_p)
df$E_r = scale(df$E_r)
# df$prev_response_bool = scale(df$prev_response_bool)

m <- glmer(subject_response_bool ~ 1 + received + E_p + E_r + prev_response_bool + 
             (1 + received + E_p + E_r + prev_response_bool | id), 
           data=df, family=binomial, 
           control=glmerControl(optimizer='optimx', 
                                optCtrl = list(method='L-BFGS-B', kkt=FALSE)))

print(summary(m))