library(lme4)
library(lmerTest)
library(dplyr)
library(simr)

# -----------
library(rstudioapi)
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path))

# --------------
path = r'(..\..\UG_data\processed_RoleChange_Study1.csv)'
df <- read.csv(path)

# ---------
df <- filter(df,
               proposerTake > 5 & 
               excluded == 0 &
               prev_prev_proposerTake > 0 & prev_subjectTake > 0 & 
               !is.na(prev_response_bool) & !is.na(subject_response_bool)
)
# -----

df$proposerTake = scale(df$proposerTake)
df$prev_prev_proposerTake = scale(df$prev_prev_proposerTake)
df$prev_subjectTake = scale(df$prev_subjectTake)


m <- glmer(subject_response_bool ~ 1 + proposerTake + prev_prev_proposerTake + prev_subjectTake + prev_response_bool + 
             (1 + proposerTake + prev_prev_proposerTake + prev_subjectTake + prev_response_bool | id), 
           data=df, family=binomial, 
           control=glmerControl(optimizer='bobyqa', optCtrl = list(maxfun = 100000)))

print(summary(m))

# ---------

p = powerSim(fit=m, test=fixed('prev_subjectTake', 'z'), nsim=10)
print(p)