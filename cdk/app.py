#!/usr/bin/env python3
import os

import aws_cdk as cdk

from cdk.langserve_stack import LangServeStack

aws_profile = os.getenv('AWS_PROFILE')
aws_account = os.getenv('CDK_DEFAULT_ACCOUNT')
aws_region = os.getenv('CDK_DEFAULT_REGION')

print(f"Using AWS profile: {aws_profile}")
print(f"Deploying to AWS account: {aws_account}")
print(f"Deploying to AWS region: {aws_region}")


app = cdk.App()

vpc_stack = app.node.try_get_context("vpc_stack") # eg: SharedVpcStack
stage = app.node.try_get_context("stage") # eg: dev
domain = app.node.try_get_context("domain") # eg: example.com , it must be there in your hosted zone
subdomain = app.node.try_get_context("subdomain") # eg: chat, if then the domains will look like chat-{stage}.domain ie chat-dev.example.com

print("stage:", stage)
print("subdomain:", subdomain)

if stage is None:
    raise ValueError("Stage is required")

env = cdk.Environment(account=aws_account, region=aws_region) # if you want some other region, you can set it or pass it similar to stage

LangServeStack(app,  f"LangServeStack-{stage}", vpc_stack, stage, subdomain, domain, env=env)

app.synth()