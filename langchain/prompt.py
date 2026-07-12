from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

class PromptTemplateManager:
    """Manage and create prompt templates."""

    def __init__(self):
        self.templates = {}

    def create_template(self, name, template_string, input_variables):
        """Create a new prompt template."""
        template = PromptTemplate(
            template=template_string,
            input_variables=input_variables
        )
        self.templates[name] = template
        return template

    def get_template(self, name):
        """Get a template by name."""
        return self.templates.get(name)

    def create_few_shot_template(self, name, examples, example_template, prefix, suffix, input_variables):
        """Create a few-shot prompt template."""
        example_prompt = PromptTemplate(
            template=example_template,
            input_variables=list(examples[0].keys())
        )

        template = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables
        )

        self.templates[name] = template
        return template

    def list_templates(self):
        """List all available templates."""
        return list(self.templates.keys())
