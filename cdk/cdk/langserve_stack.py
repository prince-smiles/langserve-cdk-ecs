from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_ecs_patterns as ecs_patterns,
    aws_certificatemanager as certificatemanager,
    aws_elasticloadbalancingv2 as elbv2,
    aws_route53 as route53,
    aws_route53_targets as targets
)
import os
from constructs import Construct
import boto3


class LangServeStack(Stack):

    def get_stack_outputs(self, stack_name):
        client = boto3.client('cloudformation', region_name=self.region)
        response = client.describe_stacks(StackName=stack_name)
        stack = response['Stacks'][0]
        outputs = stack.get('Outputs', [])
        return {output['OutputKey']: output['OutputValue'] for output in outputs}
    

    def get_vpc_details(self, vpc_stack, stage):
        shared_vpc_stack = self.get_stack_outputs(vpc_stack)
        vpc_id = shared_vpc_stack["VpcId"]

        public_subnets = [shared_vpc_stack[f"{vpc_stack}PublicSubnet{i + 1}"] for i in range(2)]
        
        public_azs = [shared_vpc_stack[f"{vpc_stack}AZPublicSubnet{i + 1}"] for i in range(2)]
        public_route_tables = [shared_vpc_stack[f"{vpc_stack}ROUTETBPublicSubnet{i + 1}"] for i in range(2)]

        if stage == "prod":
            private_subnets = [shared_vpc_stack[f"{vpc_stack}PrivateProdSubnet{i + 1}"] for i in range(2)]
            private_route_tables = [shared_vpc_stack[f"{vpc_stack}ROUTETBPrivateProdSubnet{i + 1}"] for i in range(2)]
        elif stage == "dev":
            private_subnets = [shared_vpc_stack[f"{vpc_stack}PrivateDevSubnet{i + 1}"] for i in range(2)]
            private_route_tables = [shared_vpc_stack[f"{vpc_stack}ROUTETBPrivateDevSubnet1"] for i in range(2)]
        else:
            private_subnets = [shared_vpc_stack[f"{vpc_stack}PrivateSharedSubnet{i + 1}"] for i in range(2)]
            private_route_tables = [shared_vpc_stack[f"{vpc_stack}ROUTETBPrivateSharedSubnet{i + 1}"] for i in range(2)]

        return {
            "public_subnets": public_subnets,
            "public_route_tables": public_route_tables,
            "private_subnets": private_subnets,
            "private_route_tables": private_route_tables,
            "vpc_id": vpc_id,
            "azs": public_azs,
        }

    def get_certificate_arn(self, domain_name):
        client = boto3.client('acm', region_name=self.region)
        response = client.list_certificates(CertificateStatuses=['ISSUED'])
        
        for cert in response['CertificateSummaryList']:
            if domain_name in cert['DomainName']:
                return cert['CertificateArn']
        
        raise Exception(f'Certificate not found for domain {domain_name}')


    def __init__(self, scope: Construct, construct_id: str, vpc_stack: str, stage: str, subdomain: str, 
                 domain_name: str, env=None, **kwargs) -> None:
        super().__init__(scope, construct_id, env=env, **kwargs)



        # The code that defines your stack goes here
        print("stage", stage)
        print("account", env.account)
        print("region", env.region)

        endpoint = f"{subdomain}-{stage}.{domain_name}"
        print("Endpoint", endpoint)
        vpc_details = self.get_vpc_details(vpc_stack, stage)

        print("VPC Subnets", vpc_details)

        self.vpc = ec2.Vpc.from_vpc_attributes(
            self, "VPC",
            vpc_id=vpc_details["vpc_id"],
            availability_zones=vpc_details["azs"], # Automatically fetches availability zones for the VPC
            public_subnet_ids=vpc_details["public_subnets"],
            public_subnet_route_table_ids=vpc_details["public_route_tables"],
            private_subnet_ids=vpc_details["private_subnets"],
            private_subnet_route_table_ids=vpc_details["private_route_tables"]
        )

        self.ecs_cluster = ecs.Cluster(
            self,
            "MyECSCluster",
            vpc=self.vpc,
        )

        task_role = iam.Role(
            self, "EcsTaskExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com")
        )

        # Attach policies to the role
        task_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AmazonECSTaskExecutionRolePolicy")
        )

        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "ssm:GetParameter",
                    "ssm:GetParameters",
                    "ssm:GetParametersByPath",
                    # "dynamodb:GetItem"
                ],
                resources=[
                    f"arn:aws:ssm:{env.region}:{env.account}:parameter/myapp/mongodb-connection-string", # this not used shown for demo purpose
                    # f"arn:aws:dynamodb:us-west-2:{env.account}:table/{stage}-sample-table" # if we want to give access to any dynamodb for example to the execution role
                ]
            )
        )

        image = ecs.ContainerImage.from_asset(
            directory="../chatbot",
        )

        # (v) Create ECS Task Definition
        task_definition = ecs.FargateTaskDefinition(
            self,
            "FastAPITaskDefinition",
            task_role=task_role
        )


        OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
        LANGCHAIN_TRACING_V2 = os.environ["LANGCHAIN_TRACING_V2"]
        LANGCHAIN_API_KEY = os.environ["LANGCHAIN_API_KEY"]
        LANGCHAIN_PROJECT = os.environ["LANGCHAIN_PROJECT"]
        MONGO_CONNECTION_STRING = os.environ["MONGO_CONNECTION_STRING"]
        MONGO_DATABASE = os.environ["MONGO_DATABASE"]
        MONGO_COLLECTION = os.environ["MONGO_COLLECTION"]

        container = task_definition.add_container(
            "fastapi-container",
            image=image,
            logging=ecs.LogDrivers.aws_logs(stream_prefix="fastapi"),
            environment={
                "OPENAI_API_KEY": OPENAI_API_KEY,
                "LANGCHAIN_TRACING_V2": LANGCHAIN_TRACING_V2,
                "LANGCHAIN_API_KEY": LANGCHAIN_API_KEY,
                "LANGCHAIN_PROJECT": LANGCHAIN_PROJECT,
                "MONGO_CONNECTION_STRING": MONGO_CONNECTION_STRING,
                "MONGO_DATABASE": MONGO_DATABASE,
                "MONGO_COLLECTION": MONGO_COLLECTION
            }
        )

        container.add_port_mappings(
            ecs.PortMapping(container_port=8080)
        )

        # (vi) Create Fargate Service and ALB
        self.ecs_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self,
            "FastAPIService",
            cluster=self.ecs_cluster,
            cpu=256,
            memory_limit_mib=512,
            desired_count=1,
            task_definition=task_definition,
            task_subnets=ec2.SubnetSelection(
                subnets=[ec2.Subnet.from_subnet_id(self, f"subnet-{i + 1}", subnet) for i, subnet in enumerate(vpc_details["private_subnets"])]
            ),
        )

        # (vii) Import existing certificate
        
        certificate_arn = self.get_certificate_arn(domain_name)
        print("Certificate ARN", certificate_arn)
        certificate = certificatemanager.Certificate.from_certificate_arn(
            self,
            "MyExistingCertificate",
            certificate_arn=certificate_arn
        )


        # (viii) Add HTTPS listener to the ALB
        https_listener = self.ecs_service.load_balancer.add_listener(
            "HTTPSListener",
            port=443,
            certificates=[certificate],
            default_action=elbv2.ListenerAction.forward([self.ecs_service.target_group])
        )

        # (ix) Modify the existing HTTP listener to redirect to HTTPS
        http_listener = self.ecs_service.listener
        http_listener.add_action(
            "RedirectToHTTPS",
            action=elbv2.ListenerAction.redirect(
                protocol="HTTPS",
                port="443",
                permanent=True
            )
        )

        # (x) Lookup existing hosted zone
        hosted_zone = route53.HostedZone.from_lookup(
            self,
            "HostedZone",
            domain_name=domain_name
        )

        # (xi) Create a Route 53 record to point to the ALB
        route53.ARecord(
            self,
            "AliasRecord",
            zone=hosted_zone,
            target=route53.RecordTarget.from_alias(
                targets.LoadBalancerTarget(self.ecs_service.load_balancer)
            ),
            record_name=f"{subdomain}-{stage}"
        )
        print("Service Hosted at", f"https://{endpoint}")