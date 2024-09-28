library(lme4)
library(lmerTest)
library(dplyr)

# -----------
library(rstudioapi)
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path))
# --------------

path = r'(..\..\UG_data\processed_RoleChange_Study1.csv)'
df <- read.csv(path)

# ---------
df <- filter(df,
               received < 5 & 
               excluded == 0 &
               E_p > 0 & E_r > 0 & 
               !is.na(prev_response_bool) & !is.na(subject_response_bool)
             )

df$received= scale(df$received)
df$E_p = scale(df$E_p)
df$E_r = scale(df$E_r)

m <- glmer(subject_response_bool ~ 1 + received + E_p + E_r + prev_response_bool + 
             (1 + received + E_p + E_r + prev_response_bool | id), 
           data=df, family=binomial, 
           control=glmerControl(optimizer='bobyqa', optCtrl = list(maxfun = 100000)))

print(summary(m))