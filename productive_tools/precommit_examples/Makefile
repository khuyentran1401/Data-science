all: install_format_packages download_config add_precommit install_git_hooks
install_format_packages:
	pip install flake8
	pip install isort
	pip install black
	pip install nbstripout
	pip install pre-commit

download_config:
	wget https://gist.githubusercontent.com/khuyentran1401/f4a4c822320a9db266ad935b501be66e/raw/3ebf2b029490a970591caa733dd9d2e9767b184e/.flake8
	wget https://gist.githubusercontent.com/khuyentran1401/2d6fc241f6738fdc0946e62e1597223e/raw/621d516dfc010d94ccdb8773d889f3173c996fb0/.pre-commit-config.yaml

add_precommit:
	wget -O precommit https://gist.githubusercontent.com/khuyentran1401/b86812c900110c5cf4450e183669ae48/raw/55041856a74a27245f4273040516690c856a3f0a/pyproject.toml
	touch pyproject.toml 
	cat precommit >> pyproject.toml    
	rm precommit    

install_git_hooks:   
	pre-commit install      

                                                   
