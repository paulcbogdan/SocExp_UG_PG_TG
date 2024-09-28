library(lme4)
library(lmerTest)
library(simr)
library(dplyr)

# -----------
library(rstudioapi)
current_path = rstudioapi::getActiveDocumentContext()$path 
setwd(dirname(current_path ))
# --------------

path = r'(..\..\UG_data\processed_RoleChange_Study1.csv)'
df <- read.csv(path)

# ---------


df <- filter(df,
             proposerTake > 5 & 
               excluded == 0 &
               prev_prev_proposerTake > 0 & prev_subjectTake > 0 & 
               !is.na(prev_response_bool) & !is.na(subject_response_bool) & 
               !is.na(prev_prev_subject_response_bool)
)

df$proposerTake = scale(df$proposerTake)
df$prev_prev_proposerTake = scale(df$prev_prev_proposerTake)
df$prev_subjectTake = scale(df$prev_subjectTake)

# This regresses prev_subjectTake (amount proposed in trial[n-1]) 
#     on prev_prev_proposerTake (amount partner proposed in trial[n-2])
# This is done rather than regressing trial[n] on trial[n-1] to ensure the same
#     pool of trials is being used as for the analysis of the amount proposed
#     on subsequent rejection likelihood.
# Covariates are included.
m <- lmer(prev_subjectTake ~ 1 + prev_prev_proposerTake + prev_prev_subject_response_bool + 
            (1 + prev_prev_proposerTake + prev_prev_subject_response_bool | id), 
          data=df, 
          control=lmerControl(optimizer='bobyqa', optCtrl = list(maxfun = 100000)))

print(summary(m))


# regress subject_response_bool (rejection likelihood) on:
#   prev_subjectTake (trial[n-1] proposed)
#   prev_prev_proposerTake (trial[n-2] received)
#   and covariates
m <- glmer(subject_response_bool ~ 1 + proposerTake + prev_prev_proposerTake + prev_subjectTake + prev_response_bool + 
             (1 + proposerTake + prev_prev_proposerTake + prev_subjectTake + prev_response_bool | id), 
           data=df, family=binomial, 
           control=glmerControl(optimizer='bobyqa', optCtrl = list(maxfun = 100000)))

print(summary(m))