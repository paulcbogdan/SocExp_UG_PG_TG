library(lme4)
library(lmerTest)
library(dplyr)
library(simr)

# ---- Orient current path to where this file is
library(rstudioapi)
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path))

# ---- Load data
path = r'(..\..\UG_data\processed_RoleChange_Study1.csv)'
df <- read.csv(path)

# ---- Filter (applied for needs of both the expectation and trial-trial regresssions)
print('--***--*****--***-- | expectations | --***--*****--***--')
df <- filter(df,
             received < 5 & 
               excluded == 0 &
               E_p > 0 & E_r > 0 & 
               prev_prev_proposerTake > 0 & prev_subjectTake > 0 & 
               !is.na(prev_response_bool) & !is.na(subject_response_bool)
)

# ---- Fit lmer 
df$received= scale(df$received)
df$E_p = scale(df$E_p)
df$E_r = scale(df$E_r)

m <- glmer(subject_response_bool ~ 1 + received + E_p + E_r + prev_response_bool + 
             (1 + received + E_p + E_r + prev_response_bool | id), 
           data=df, family=binomial, 
           control=glmerControl(optimizer='bobyqa', optCtrl = list(maxfun = 100000)))

print(summary(m))

# ----
fixef(m)['E_p'] = 0.50
p = powerSim(fit=m, test=fixed('E_p', 'z'), nsim=300)
print(p)

# -------------------------------------
print('--***--*****--***-- | trial-trial | --***--*****--***--')

df$proposerTake = scale(df$proposerTake)
df$prev_prev_proposerTake = scale(df$prev_prev_proposerTake)
df$prev_subjectTake = scale(df$prev_subjectTake)

m2 <- glmer(subject_response_bool ~ 1 + proposerTake + prev_prev_proposerTake + prev_subjectTake + prev_response_bool + 
             (1 + proposerTake + prev_prev_proposerTake + prev_subjectTake + prev_response_bool | id), 
           data=df, family=binomial, 
           control=glmerControl(optimizer='bobyqa', optCtrl = list(maxfun = 100000)))

print(summary(m2))

# ----
fixef(m2)['prev_subjectTake'] = 0.3

p2 = powerSim(fit=m2, test=fixed('prev_subjectTake', 'z'), nsim=1000)
print(p2)