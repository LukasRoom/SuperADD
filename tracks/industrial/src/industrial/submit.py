from industrial.paths import get_submission_path, get_results_path
from industrial.submission import prepare_submission, validate_submission

def main() -> None:
    print(f'validating submission in directory {get_results_path()}')
    valid = validate_submission(get_results_path())
    print(f'status: {"VALID" if valid else "INVALID"}')

    if not valid:
        exit(1)

    print(f'writing to file {get_submission_path()}')
    prepare_submission(get_results_path(), get_submission_path())
    print(f'finished')

if __name__ == "__main__":
    main()
