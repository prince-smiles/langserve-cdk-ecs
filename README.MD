# Overview

This shows a Basic Project where we host a Langchain Agent Chatbot with history powered by MongoDB.


# Prerequisite

Please deploy the Shared VPC Stack given here https://github.com/mathlover777/shared-vpc

Please create a mongoDB project and get a connection string. You can make it accessible from all IP addresses. But if you want to restrict it, you can get IP address associated with the Nat Gateway of the VPC, as all tasks will run in private subnets and they will be exposed to outside world by one single natgateway shared accross all stages.


Also get a Langsmith account and we will be using it for tracing. Also if you have multiple AWS profiles, then how to export the AWS_PROFILE is mentioned here https://github.com/mathlover777/shared-vpc


# How to create a project like this from scratch

1. Create a Empty directory (the root directoy and get inside that) and make sure you have langchain-cli installed in some python development.
You can use this [Langserve](https://python.langchain.com/v0.1/docs/langserve/), for actual code we will use venv inside, so just use some environment to bootstrap the project.
2. Initialze the langserve project ->
```shell
langchain app new chatbot
```
3. Install the 3rd party packages as `poetry add langchain-openai`, here I have added everything I needed and hence they are available in the poetry file.
4. Setup `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY` and `LANGCHAIN_PROJECT` if You plan to use Langsmith [Ref](https://docs.smith.langchain.com/old/cookbook/tracing-examples/runnable-naming). Here `LANGCHAIN_PROJECT` will be automatically set by next step, it needs not be explictly created in Langsmith.

5. Refer to the `.env.sample` and create a similar as `.env` that will be used in production.

Also if you want to overwrite some of them in local testing then keep them in `.env.local` (similar for `.env.local.sample`)

If you want to quickly set them in your shell from the file, you can run
```shell
export $(cat .env | xargs)
```

6. Now we setup the poetry environment as follows and use the Poetry Python as VS Codes Interpreter so that we can debug code properly
```shell
cd chatbot
poetry install
poetry env use /opt/homebrew/bin/python3.11 # you need to use some poetry environment recommended > 3.11, this will be done for ocal testing
poetry install
source $(poetry env info --path)/bin/activate
```

Ideally it will install all dependencies for the project, but we can add anything like this
```shell
poetry add langchain
poetry add langchain_openai
poetry add langchain_mongodb
```
These are already added, and should be installed by `poetry install`

Also at this time, change visual studios python interpreter to the poetry one. You can get its path (assuming the shell is activated properly)
```shell
❯ which python
/Users/sourav/Library/Caches/pypoetry/virtualenvs/chatbot-Aez_kRcH-py3.11/bin/python
```
**NOTE** Please make sure in both terminal and VS interpreter the same poetry python is set, else you will get unpredictable results.
I have changed the `chatbot.app.server.py` code to a basic agent, so it can be run now by this
```shell
langchain serve
```

7. Change the `Dockerfile`'s from to this `FROM  --platform=linux/amd64 python:3.11-slim` (to tell the final image will be run on Linux system ie fargate)
8. Create the `docker-compose.yml` file as mentioned.
9. We need to have the AWS credentials in Docker. In ECS, the task will have necessary IAM permissions. But if we want to test locally, we need our local AWS Credentials
```shell
docker volume create aws-config
docker run --rm --platform linux/arm64 -v aws-config:/root/.aws -v ~/.aws:/aws alpine sh -c "mkdir -p /root/.aws && cp -r /aws/* /root/.aws"
```
To run it 
```shell
docker-compose up --build
```
Earlier in the `.env` file we mentioned `AWS_PROFILE`. So even if we copy all the credentials, as this is set as Environment variable inside docker,
so if we use any AWS Api call using `boto3`, it will know to selcect credentials and region for which profile.
10. Setup CDK [CDK AWS Documentation](https://docs.aws.amazon.com/cdk/v2/guide/cli.html), in the root run this
```shell
mkdir cdk
cd cdk  
cdk init app --language python --generate-only
```
11. In the same environment now install the cdk libraries mentioned in `root/cdk/requirements.txt`. You can create a new one also, but then in VS Code you need to keep on switching the interpreter.
```shell
pip install --upgrade -r requirements.txt 
```
12. Change the code in CDK. It will create some file and class as `cdk_stack.py` and `CdkStack`. You can change them to whatever values. Note they are imported in various places like `cdk/app.py`, so change there also like a normal Python Project.
13. The AWS Profile mentioned `.env` file is only for Docker. For CDK Deployment and non docker local testing you need to set it explicitly
```shell
export AWS_PROFILE=your profile
```
Make sure its properly set, run 
```shell
cdk synth --context stage=dev
```
it should print the account id and the default region associated with the profile.
14. Set the environment variables needed for deployment, run this in root
```shell
export $(cat .env | xargs)
```
(Assuming you have kept them in the .env file in root)

15. Deploy: Finally you can deploy as 
```shell
cdk deploy --context stage=dev  
```