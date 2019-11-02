<?php
	function call_python() {
		$output = exec('python test.py');
		echo $output;
	}

	if ($_FILES["file"]["error"] > 0) {
		echo "Error: " . $_FILES["file"]["error"];
	}
	else {
		echo "file_name: " . $_FILES["file"]["name"]."<br/>";
		echo "file_type: " . $_FILES["file"]["type"]."<br/>";
		echo "file capity: " . ($_FILES["file"]["size"] / 1024)." Kb<br />";
		echo "temp_file_name: " . $_FILES["file"]["tmp_name"];

		if (file_exists("./upload/" . $_FILES["file"]["name"])) {
			echo "file is exist, do not repeated~";

		}
		else {
			move_uploaded_file($_FILES["file"]["tmp_name"],"./upload/".$_FILES["file"]["name"]);
		}
	}

?>
